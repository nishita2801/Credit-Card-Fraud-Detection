
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import prepare_data


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_models():
    """
    Define the two classification models.
    """

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
                solver="lbfgs"
            ))
        ]),

        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ])

    }

    return models


def train_models():
    """
    Train both classifiers and save trained models.
    """

    print("\n========== MODEL TRAINING ==========")

    # Load prepared data
    X_train, X_test, y_train, y_test = prepare_data()

    # Build classifiers
    models = build_models()

    trained_models = {}

    # Train each model
    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        trained_models[model_name] = model

        # Save individual model
        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(model, model_path)

        print(f"{model_name} trained successfully!")
        print(f"Model saved: {model_path}")

    # Save models and test data for evaluation
    artifacts = {
        "models": trained_models,
        "X_test": X_test,
        "y_test": y_test
    }

    artifact_path = MODEL_DIR / "evaluation_artifacts.joblib"

    joblib.dump(artifacts, artifact_path)

    print("\n========== TRAINING SUMMARY ==========")

    print("Models trained:", len(trained_models))

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nSaved model files:")

    for file in MODEL_DIR.glob("*.joblib"):
        print("-", file.name)

    print("\nBoth models trained and saved successfully!")

    return trained_models


if __name__ == "__main__":

    train_models()

