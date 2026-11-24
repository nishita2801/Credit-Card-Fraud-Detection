
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset


RANDOM_STATE = 42
TEST_SIZE = 0.20


def prepare_data():
    """
    Clean the dataset and split it into training and testing sets.

    Exact duplicate rows are removed before splitting.
    Stratification preserves the class distribution.
    """

    # Step 1: Load original dataset
    df = load_dataset()

    print("\n========== DATA PREPROCESSING ==========")

    print(f"Original dataset shape: {df.shape}")

    # Step 2: Remove exact duplicate records
    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Duplicate records removed: {duplicate_count}")
    print(f"Cleaned dataset shape: {df.shape}")

    # Step 3: Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Step 4: Check class distribution
    print("\nCleaned class distribution:")
    print(y.value_counts())

    print("\nCleaned class percentages:")
    print((y.value_counts(normalize=True) * 100).round(4))

    # Step 5: Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Step 6: Display split information
    print("\n========== TRAIN-TEST SPLIT ==========")

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    print("\nPreprocessing completed successfully!")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    print("\nFinal verification:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
