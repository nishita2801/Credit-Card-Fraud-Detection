
from pathlib import Path
import pandas as pd


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def load_dataset():
    """
    Load the original credit card fraud dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}\n"
            "Please download creditcard.csv into data/raw/"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"Amount", "Class"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{required_columns - set(df.columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class column must contain binary labels 0 and 1.")

    if df["Class"].isna().any():
        raise ValueError("Target column contains missing values.")

    return df


def display_dataset_summary(df):
    """
    Display important dataset information.
    """

    print("\n========== DATASET SUMMARY ==========")

    print(f"\nTotal transactions: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

    print("\nClass percentages:")
    print(
        (df["Class"].value_counts(normalize=True) * 100)
        .round(4)
    )

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFirst five records:")
    print(df.head())


if __name__ == "__main__":

    dataset = load_dataset()

    display_dataset_summary(dataset)

    print("\nDataset loaded successfully!")
