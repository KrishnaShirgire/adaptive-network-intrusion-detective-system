
from pathlib import Path
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from preprocessing import (
    load_data,
    prepare_features,
    build_preprocessor,
)


# Create output folders
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# 1. Load the official training and testing datasets
train_df, test_df = load_data()

X_train, y_train = prepare_features(train_df)
X_test, y_test = prepare_features(test_df)


# 2. Load the features selected by the Genetic Algorithm
selected_features = pd.read_csv(
    "results/ga_selected_features.csv"
)["selected_feature"].tolist()

# Ensure the selected features exist in both datasets
missing_train = set(selected_features) - set(X_train.columns)
missing_test = set(selected_features) - set(X_test.columns)

if missing_train or missing_test:
    raise ValueError(
        f"Missing training features: {missing_train}; "
        f"missing testing features: {missing_test}"
    )

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print("Original feature count:", X_train.shape[1])
print("GA-selected feature count:", len(selected_features))
print("Selected features:", selected_features)


# 3. Build the optimized Random Forest pipeline
optimized_model = Pipeline([
    ("preprocessor", build_preprocessor(X_train_selected)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])


# 4. Train using the full official training dataset
print("\nTraining optimized Random Forest...")
optimized_model.fit(X_train_selected, y_train)
print("Training completed!")


# 5. Predict on the official testing dataset
start_time = time.perf_counter()
y_pred = optimized_model.predict(X_test_selected)
prediction_time = time.perf_counter() - start_time


# 6. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
).ravel()

false_positive_rate = (
    fp / (fp + tn) if (fp + tn) else 0.0
)

print("\n===== OPTIMIZED RANDOM FOREST RESULTS =====")
print(f"Accuracy           : {accuracy:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"F1-score           : {f1:.4f}")
print(f"False-positive rate: {false_positive_rate:.4f}")
print(f"Prediction time    : {prediction_time:.4f} seconds")

print("\nConfusion Matrix:")
print([[tn, fp], [fn, tp]])


# 7. Save the optimized model
joblib.dump(
    optimized_model,
    "models/optimized_rf.joblib",
)
print("\nSaved model to models/optimized_rf.joblib")


# 8. Compare with the saved baseline model
baseline_model = joblib.load("models/baseline_rf.joblib")

start_time = time.perf_counter()
baseline_predictions = baseline_model.predict(X_test)
baseline_prediction_time = time.perf_counter() - start_time

baseline_tn, baseline_fp, baseline_fn, baseline_tp = (
    confusion_matrix(
        y_test, baseline_predictions, labels=[0, 1]
    ).ravel()
)

baseline_fpr = (
    baseline_fp / (baseline_fp + baseline_tn)
    if (baseline_fp + baseline_tn)
    else 0.0
)

comparison = pd.DataFrame([
    {
        "Model": "Baseline Random Forest",
        "Original_features": X_train.shape[1],
        "Selected_features": X_train.shape[1],
        "Accuracy": accuracy_score(y_test, baseline_predictions),
        "Precision": precision_score(
            y_test, baseline_predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, baseline_predictions, zero_division=0
        ),
        "F1_score": f1_score(
            y_test, baseline_predictions, zero_division=0
        ),
        "False_positive_rate": baseline_fpr,
        "Prediction_time_seconds": baseline_prediction_time,
    },
    {
        "Model": "GA-Optimized Random Forest",
        "Original_features": X_train.shape[1],
        "Selected_features": len(selected_features),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_score": f1,
        "False_positive_rate": false_positive_rate,
        "Prediction_time_seconds": prediction_time,
    },
])

comparison.to_csv(
    "results/model_comparison.csv",
    index=False,
)

print("\n===== BASELINE VS OPTIMIZED =====")
print(comparison.to_string(index=False))
print("\nSaved comparison to results/model_comparison.csv")
