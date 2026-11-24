import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.visualizations import (
    create_confusion_matrix,
    create_roc_curve,
    create_metrics_table,
    create_class_distribution_chart
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# LOAD CUSTOM CSS THEME
# ==========================================

def load_custom_css():
    """Load the optional dashboard theme from assets/dashboard.css."""
    css_path = Path(__file__).resolve().parent / "assets" / "dashboard.css"

    try:
        css = css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning(
            "Theme file not found. Create assets/dashboard.css "
            "to apply the custom dashboard styling."
        )
        return

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_custom_css()


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RESULTS_PATH = MODEL_DIR / "evaluation_results.joblib"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
SUMMARY_PATH = REPORT_DIR / "dataset_summary.json"


# ==========================================
# LOAD EVALUATION RESULTS
# ==========================================

@st.cache_resource
def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Evaluation results not found. "
            "Make sure models/evaluation_results.joblib is available."
        )

    return joblib.load(RESULTS_PATH)


@st.cache_data
def load_comparison():
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Comparison report not found. "
            "Make sure reports/model_comparison.csv is available."
        )

    return pd.read_csv(COMPARISON_PATH, index_col=0)


@st.cache_data
def load_dataset_summary():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Dataset summary not found. "
            "Make sure reports/dataset_summary.json is available."
        )

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    class_counts = {
        int(label): int(count)
        for label, count in summary["class_counts"].items()
    }

    total_transactions = int(summary["total_transactions"])

    return total_transactions, class_counts


# ==========================================
# HEADER
# ==========================================

st.title("💳 Credit Card Fraud Detection")
st.subheader("Machine Learning Model Evaluation Dashboard")

st.markdown(
    """
    Compare machine learning classifiers for detecting fraudulent
    credit card transactions using real-world data.
    """
)

st.divider()


# ==========================================
# LOAD DATA
# ==========================================

try:
    results = load_results()
    comparison_df = load_comparison()
    total_transactions, class_counts = load_dataset_summary()

except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(str(error))
    st.info(
        "Please make sure the evaluation results, comparison report, "
        "and dataset summary files are available."
    )
    st.stop()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Dashboard Controls")

model_names = list(results.keys())

selected_model = st.sidebar.selectbox(
    "Select Model for Confusion Matrix",
    model_names
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")
st.sidebar.write("Dataset: Credit Card Fraud Detection")
st.sidebar.write("Models: Logistic Regression, Random Forest")
st.sidebar.write("Training: 80%")
st.sidebar.write("Testing: 20%")


# ==========================================
# OVERVIEW METRICS
# ==========================================

st.header("1. Dataset Overview")

total_fraud = int(class_counts.get(1, 0))
total_legitimate = int(class_counts.get(0, 0))

fraud_percentage = (
    total_fraud / total_transactions * 100
    if total_transactions
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Legitimate Transactions",
    f"{total_legitimate:,}"
)

col3.metric(
    "Fraudulent Transactions",
    f"{total_fraud:,}"
)

col4.metric(
    "Fraud Percentage",
    f"{fraud_percentage:.4f}%"
)

st.plotly_chart(
    create_class_distribution_chart(class_counts),
    use_container_width=True
)


# ==========================================
# MODEL COMPARISON TABLE
# ==========================================

st.divider()

st.header("2. Model Performance Comparison")

st.markdown(
    "Compare both classifiers using the required evaluation metrics."
)

display_df = create_metrics_table(comparison_df)

st.dataframe(
    display_df,
    use_container_width=True
)

st.caption(
    "All evaluation metrics are calculated on the same untouched "
    "stratified test dataset."
)


# ==========================================
# MODEL PERFORMANCE CARDS
# ==========================================

st.subheader("Individual Model Performance")

columns = st.columns(len(model_names))

for column, model_name in zip(columns, model_names):
    metrics = results[model_name]

    with column:
        st.markdown(f"### {model_name}")

        st.metric(
            "Precision",
            f"{metrics['Precision']:.4f}"
        )

        st.metric(
            "Recall",
            f"{metrics['Recall']:.4f}"
        )

        st.metric(
            "F1-score",
            f"{metrics['F1-score']:.4f}"
        )

        st.metric(
            "ROC-AUC",
            f"{metrics['ROC-AUC']:.4f}"
        )


# ==========================================
# CONFUSION MATRIX
# ==========================================

st.divider()

st.header("3. Confusion Matrix Heatmap")

st.markdown(
    """
    The confusion matrix shows the number of correctly and incorrectly
    classified legitimate and fraudulent transactions.
    """
)

st.plotly_chart(
    create_confusion_matrix(results, selected_model),
    use_container_width=True
)

cm = results[selected_model]["Confusion Matrix"]
tn, fp, fn, tp = cm.ravel()

c1, c2, c3, c4 = st.columns(4)

c1.metric("True Negatives", f"{tn:,}")
c2.metric("False Positives", f"{fp:,}")
c3.metric("False Negatives", f"{fn:,}")
c4.metric("True Positives", f"{tp:,}")


# ==========================================
# ROC CURVE
# ==========================================

st.divider()

st.header("4. ROC Curve Comparison")

st.markdown(
    """
    ROC curves compare the true positive rate against the false positive
    rate across different classification thresholds.
    """
)

st.plotly_chart(
    create_roc_curve(results),
    use_container_width=True
)


# ==========================================
# BUSINESS ANALYSIS
# ==========================================

st.divider()

st.header("5. Model Selection and Business Analysis")

best_recall = max(
    model_names,
    key=lambda name: results[name]["Recall"]
)

best_f1 = max(
    model_names,
    key=lambda name: results[name]["F1-score"]
)

best_auc = max(
    model_names,
    key=lambda name: results[name]["ROC-AUC"]
)

st.markdown("### Performance Highlights")

st.write(f"**Highest Recall:** {best_recall}")
st.write(f"**Highest F1-score:** {best_f1}")
st.write(f"**Highest ROC-AUC:** {best_auc}")

st.markdown("### Business Interpretation")

st.write(
    f"""
    In credit card fraud detection, recall is important because it measures
    how many actual fraudulent transactions are successfully identified.

    The model with the highest recall is **{best_recall}**.

    Precision is also important because false fraud alerts can inconvenience
    legitimate customers.

    The model with the highest F1-score is **{best_f1}**.

    The model with the highest ROC-AUC is **{best_auc}**.

    The final model selection should consider the business cost of missed
    fraud, false alarms, and the required operating threshold.
    """
)

st.warning(
    "A model with the highest recall or ROC-AUC is not automatically "
    "the best production model. The business must consider both "
    "false positives and false negatives."
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    **Project:** Credit Card Fraud Detection – Model Evaluation Dashboard

    **Technology:** Python | Scikit-learn | Streamlit | Plotly

    **Purpose:** Evaluate and compare classification models on an
    imbalanced real-world dataset.
    """
)

st.success("Model evaluation dashboard loaded successfully!")