import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>
.main {background:#f7f9fc;}
.hero {
    padding:1.5rem 1.8rem; border-radius:18px;
    background:linear-gradient(120deg,#102a43,#2563a6);
    color:white; margin-bottom:1.2rem;
}
.hero h1 {color:white; margin-bottom:.35rem;}
.hero p {color:#e6f1ff; margin-bottom:0;}
div[data-testid="stMetric"] {
    background:white; border:1px solid #e5eaf1;
    padding:14px; border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏦 Loan Default Prediction</h1>
<p>Naïve Bayes classification · Explore the dataset, evaluate the model, and test a loan applicant.</p>
</div>
""", unsafe_allow_html=True)

TARGET = "loan_status"

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists("loan_data.csv"):
        return pd.read_csv("loan_data.csv")
    return None

@st.cache_resource(show_spinner=False)
def build_and_train(data):
    data = data.copy()
    data = data.dropna(subset=[TARGET])
    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    # Ensure the target is numeric if the CSV stores 0/1 as text.
    if y.dtype == "object":
        numeric_y = pd.to_numeric(y.astype(str).str.strip(), errors="coerce")
        if numeric_y.notna().all():
            y = numeric_y.astype(int)
        else:
            y = y.astype(str).str.strip()

    if y.nunique() < 2:
        raise ValueError("loan_status mein kam se kam 2 classes honi chahiye.")
    if len(data) < 10:
        raise ValueError("Dataset bahut chhota hai. Kam se kam 10 valid rows chahiye.")

    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    remaining = [c for c in X.columns if c not in categorical + numeric]
    categorical += remaining
    for col in categorical:
        X[col] = X[col].astype("object")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")

    # GaussianNB requires dense numeric input; OneHotEncoder above outputs dense data.
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = list(model.named_steps["classifier"].classes_)

    positive_index = next(
        (classes.index(v) for v in [1, "1", "Default", "default", "Yes", "yes"] if v in classes),
        len(classes) - 1
    )
    auc = None
    if len(classes) == 2:
        try:
            binary_actual = (y_test == classes[positive_index]).astype(int)
            auc = roc_auc_score(binary_actual, probabilities[:, positive_index])
        except Exception:
            auc = None

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classes": classes,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "positive_index": positive_index,
        "auc": auc,
        "numeric": numeric,
        "categorical": categorical,
        "features": list(X.columns),
    }
    return model, results, data

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload loan_data.csv", type=["csv"])
    st.caption("CSV mein target column ka naam `loan_status` hona chahiye.")
    st.caption("Notebook convention: 0 = No Default, 1 = Default.")

df = load_data(uploaded_file)

if df is None:
    st.info("Apni `loan_data.csv` file sidebar se upload karein, ya `app.py` ke same folder mein rakhein.")
    st.code("project-folder/\\n├── app.py\\n├── requirements.txt\\n└── loan_data.csv")
    st.stop()

if TARGET not in df.columns:
    st.error(f"`{TARGET}` column nahi mila. CSV columns: {', '.join(map(str, df.columns))}")
    st.stop()

try:
    with st.spinner("Naïve Bayes model train ho raha hai..."):
        model, result, clean_df = build_and_train(df)
except Exception as exc:
    st.error(f"Model train nahi ho paya: {exc}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🔎 Data Analysis", "🧠 Predict Applicant", "ℹ️ About Model"
])

with tab1:
    st.subheader("Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Total rows", f"{len(clean_df):,}")
    b.metric("Features", len(clean_df.columns) - 1)
    c.metric("Model accuracy", f"{result['accuracy'] * 100:.2f}%")
    d.metric("ROC-AUC", f"{result['auc']:.3f}" if result["auc"] is not None else "N/A")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Dataset preview")
        st.dataframe(clean_df.head(10), use_container_width=True)
    with right:
        st.markdown("#### Loan status distribution")
        counts = clean_df[TARGET].astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_xlabel("loan_status")
        ax.set_ylabel("Number of applicants")
        ax.set_title("Loan Status Counts")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Missing values")
    missing = clean_df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing.rename("missing_count").to_frame(), use_container_width=True)

with tab2:
    st.subheader("Exploratory Data Analysis")
    st.markdown("#### Numeric summary")
    st.dataframe(clean_df.describe(include="all").T, use_container_width=True)

    numeric_df = clean_df.select_dtypes(include=np.number)
    if len(numeric_df.columns) >= 2:
        st.markdown("#### Correlation heatmap (numeric columns)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Correlation heatmap ke liye kam se kam 2 numeric columns chahiye.")

    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(result["y_test"], result["y_pred"], labels=result["classes"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(x) for x in result["classes"]],
                yticklabels=[str(x) for x in result["classes"]], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Classification report")
    report = classification_report(
        result["y_test"], result["y_pred"], output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    if result["auc"] is not None and len(result["classes"]) == 2:
        y_binary = (result["y_test"] == result["classes"][result["positive_index"]]).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, result["probabilities"][:, result["positive_index"]])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {result['auc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

with tab3:
    st.subheader("Test a new loan applicant")
    st.write("Neeche applicant ki details bharein. Model trained dataset ke features use karega.")
    feature_cols = result["features"]
    defaults = {
        "person_age": 30, "person_gender": "male", "person_education": "Bachelor",
        "person_income": 55000, "person_emp_exp": 5,
        "person_home_ownership": "RENT", "loan_amnt": 10000,
        "loan_intent": "EDUCATION", "loan_int_rate": 12.5,
        "loan_percent_income": 0.18, "cb_person_cred_hist_length": 4,
        "credit_score": 680, "previous_loan_defaults_on_file": "No"
    }

    with st.form("applicant_form"):
        inputs = {}
        columns = st.columns(2)
        for i, col in enumerate(feature_cols):
            series = clean_df[col]
            with columns[i % 2]:
                if col in result["categorical"] or series.dtype == "object" or str(series.dtype) == "category":
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if not choices:
                        choices = ["Unknown"]
                    preferred = str(defaults.get(col, choices[0]))
                    index = choices.index(preferred) if preferred in choices else 0
                    inputs[col] = st.selectbox(col, choices, index=index)
                else:
                    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                    default = defaults.get(col, float(numeric_series.median()) if not numeric_series.empty else 0.0)
                    min_value = float(numeric_series.min()) if not numeric_series.empty else 0.0
                    max_value = float(numeric_series.max()) if not numeric_series.empty else max(float(default), 1.0)
                    if min_value == max_value:
                        inputs[col] = st.number_input(col, value=float(default))
                    else:
                        inputs[col] = st.number_input(
                            col, min_value=min_value, max_value=max_value,
                            value=float(min(max(default, min_value), max_value))
                        )
        submitted = st.form_submit_button("Predict loan status", type="primary")

    if submitted:
        applicant = pd.DataFrame([inputs], columns=feature_cols)
        for col in result["categorical"]:
            if col in applicant.columns:
                applicant[col] = applicant[col].astype("object")
        prediction = model.predict(applicant)[0]
        probabilities = model.predict_proba(applicant)[0]
        classes = result["classes"]
        positive_index = result["positive_index"]
        default_label = classes[positive_index]
        default_probability = float(probabilities[positive_index])
        is_default = prediction == default_label

        if is_default:
            st.error("⚠️ Model prediction: LOAN DEFAULT")
        else:
            st.success("✅ Model prediction: NO DEFAULT")

        x, y = st.columns(2)
        x.metric("Predicted class", str(prediction))
        y.metric("Predicted class probability", f"{max(probabilities) * 100:.1f}%")
        st.write("Class probabilities")
        prob_df = pd.DataFrame({
            "Class": [str(cls) for cls in classes],
            "Probability": probabilities
        })
        st.dataframe(prob_df, use_container_width=True)
        st.caption("Educational project only — prediction ko real lending decision ke liye akela use na karein.")

with tab4:
    st.subheader("About this project")
    st.markdown("""
**Algorithm:** Gaussian Naïve Bayes (`GaussianNB`)

**Target:** `loan_status`
- `0` = No Default
- `1` = Default

**Workflow**
1. Load the CSV dataset.
2. Inspect rows, columns, missing values, and distributions.
3. Impute missing numeric/categorical values and encode categories.
4. Split data into 80% training and 20% testing (random state 42).
5. Train the Gaussian Naïve Bayes classifier.
6. Evaluate accuracy, classification report, confusion matrix, and ROC-AUC.

**Important:** This app is a learning/demo project. Model predictions can be wrong and should not be used as the sole basis for approving or rejecting real loan applications.
""")
