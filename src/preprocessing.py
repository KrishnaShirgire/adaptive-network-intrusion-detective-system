
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer #ColumnTransformer applies the appropriate preprocessing to each type of column.
from sklearn.impute import SimpleImputer  #SimpleImputer fills missing values if any appear in future data
from sklearn.pipeline import Pipeline  #Connects multiple processing steps in order.
from sklearn.preprocessing import OneHotEncoder #OneHotEncoder converts text categories such as protocols and services into numerical features.


# Dataset paths
TRAIN_PATH = Path("data/raw/training-set.csv")
TEST_PATH = Path("data/raw/testing-set.csv")


def load_data():  #reads both CSV files.
    """Load training and testing datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Remove accidental spaces from column names
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    return train_df, test_df


def prepare_features(df):  #separates the traffic features from the answer (label)
    """Separate model inputs (X) from the target (y)."""
    X = df.drop(columns=["label", "attack_cat", "id"])
    y = df["label"]

    return X, y


def build_preprocessor(X):
    """Build preprocessing steps for numeric and text features."""
    numeric_columns = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))  #SimpleImputer fills missing values if any appear in future data
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return preprocessor


if __name__ == "__main__":
    train_df, test_df = load_data()

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor(X_train)

    # Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train) #fit_transform() learns the preprocessing rules from training data.

    # Apply the same fitted preprocessing to test data
    X_test_processed = preprocessor.transform(X_test) #transform() applies those same rules(from fit_transform) to testing data without learning from the test set.

    print("Training dataset shape:", train_df.shape)
    print("Testing dataset shape:", test_df.shape)
    print("Number of input features before encoding:", X_train.shape[1])
    print("Numeric columns:", len(
        X_train.select_dtypes(include=["number", "bool"]).columns
    ))
    print("Categorical columns:", len(
        X_train.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    ))
    print("Processed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)
    print("Training labels:", y_train.shape)
    print("Testing labels:", y_test.shape)
    print("\nPreprocessing completed successfully!")
