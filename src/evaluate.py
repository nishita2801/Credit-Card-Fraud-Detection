
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models():
    """
    Evaluate trained classifiers using:
    Precision, Recall, F1-score and ROC-AUC.

    Also generate confusion matrices and ROC curve data.
    """

    print("\n========== MODEL EVALUATION ==========")

    # Load saved training artifacts
    artifact_path = MODEL_DIR / "evaluation_artifacts.joblib"

    if not artifact_path.exists():
        raise FileNotFoundError(
            "Evaluation artifacts not found. Run python -m src.train first."
        )

    artifacts = joblib.load(artifact_path)

    models = artifacts["models"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    results = {}

    # Evaluate each classifier
    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)

        # Fraud probabilities
        y_probability = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test, y_probability
        )

        accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        # ROC curve coordinates
        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_probability
        )

        # Store all evaluation results
        results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "Confusion Matrix": cm,
            "FPR": fpr,
            "TPR": tpr,
            "Thresholds": thresholds,
            "Predictions": y_pred,
            "Probabilities": y_probability,
            "Classification Report": classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        }

        # Print model metrics
        print("\nPerformance Metrics:")

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(results[model_name]["Classification Report"])

    # Create comparison table
    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ]

    comparison_data = {
        model_name: {
            metric: results[model_name][metric]
            for metric in metrics
        }
        for model_name in results
    }

    comparison_df = pd.DataFrame(comparison_data).T

    # Save comparison table
    comparison_path = REPORT_DIR / "model_comparison.csv"

    comparison_df.to_csv(comparison_path)

    # Save complete evaluation results
    evaluation_path = MODEL_DIR / "evaluation_results.joblib"

    joblib.dump(results, evaluation_path)

    # Display final comparison
    print("\n========== MODEL COMPARISON ==========")

    print(comparison_df.round(4))

    print("\nEvaluation files saved:")
    print(comparison_path)
    print(evaluation_path)

    print("\nModel evaluation completed successfully!")

    return results, comparison_df


if __name__ == "__main__":

    evaluate_models()
