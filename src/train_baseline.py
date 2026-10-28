
from pathlib import Path

import joblib  # Saves the trained model to a file.
from sklearn.ensemble import RandomForestClassifier  #Implements the Random Forest algorithm.
from sklearn.metrics import (  #Measures model performance.
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline  #  Connects preprocessing and model training into one workflow.

from preprocessing import (
    load_data,
    prepare_features,
    build_preprocessor,
)


# Create output folders if they do not exist
Path("models").mkdir(exist_ok=True)    # models stores the trained model.
Path("results").mkdir(exist_ok=True)


# 1. Load datasets
train_df, test_df = load_data()

X_train, y_train = prepare_features(train_df)
X_test, y_test = prepare_features(test_df)


# 2. Build the complete pipeline
model = Pipeline([
    ("preprocessor", build_preprocessor(X_train)),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])


# 3. Train the model
print("Training baseline Random Forest...")
model.fit(X_train, y_train)
print("Training completed!")


# 4. Predict labels for the test dataset
y_pred = model.predict(X_test)


# 5. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n===== BASELINE RANDOM FOREST RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Attack"],
    zero_division=0,
))


# 6. Save the complete pipeline
joblib.dump(model, "models/baseline_rf.joblib")

print("Saved model to models/baseline_rf.joblib")


# Accuracy: percentage of all predictions that are correct.
# Precision: how many predicted attacks were actually attacks.
# Recall: how many actual attacks the model detected.
# F1-score: balance between precision and recall.
# Confusion matrix: shows correct and incorrect normal/attack predictions.