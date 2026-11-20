
from pathlib import Path
from io import BytesIO
import uuid
import os

import joblib
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    url_for,
)


# --------------------------------------------------
# 1. Initialize Flask
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# --------------------------------------------------
# 2. Define project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

BASELINE_PATH = MODEL_DIR / "baseline_rf.joblib"
OPTIMIZED_PATH = MODEL_DIR / "optimized_rf.joblib"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"


# --------------------------------------------------
# 3. Load trained models and comparison results
# --------------------------------------------------
baseline_model = joblib.load(BASELINE_PATH)
optimized_model = joblib.load(OPTIMIZED_PATH)

comparison_df = pd.read_csv(COMPARISON_PATH)

MODELS = {
    "baseline": baseline_model,
    "optimized": optimized_model,
}

MODEL_NAMES = {
    "baseline": "Baseline Random Forest",
    "optimized": "GA-Optimized Random Forest",
}

# Temporary storage for downloadable prediction CSVs.
# The CSV contents stay on the server, not in the browser cookie.
download_files = {}


# --------------------------------------------------
# 4. Main dashboard page
# --------------------------------------------------
@app.route("/")
def dashboard():
    comparison = comparison_df.to_dict(orient="records")

    return render_template(
        "index.html",
        comparison=comparison,
        error=None,
        result=None,
        selected_model="optimized",
    )


# --------------------------------------------------
# 5. Upload CSV and predict
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")
    selected_model = request.form.get("model", "optimized")

    if selected_model not in MODELS:
        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error="Please select a valid model.",
            result=None,
            selected_model="optimized",
        )

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error="Please choose a CSV file first.",
            result=None,
            selected_model=selected_model,
        )

    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error="Only CSV files are supported.",
            result=None,
            selected_model=selected_model,
        )

    try:
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.astype(str).str.strip()

        if data.empty:
            raise ValueError("The uploaded CSV contains no records.")

        # Prevent ambiguous duplicate column names.
        if data.columns.duplicated().any():
            raise ValueError(
                "The CSV contains duplicate column names."
            )

        # Remove target and ID columns from model inputs.
        # If 'label' is present, it is NOT used for prediction.
        features = data.drop(
            columns=["label", "attack_cat", "id"],
            errors="ignore",
        )

        model = MODELS[selected_model]

        # Get the exact raw input columns expected by this model.
        required_columns = list(
            model.named_steps["preprocessor"].feature_names_in_
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in features.columns
        ]

        if missing_columns:
            preview = ", ".join(missing_columns[:10])
            if len(missing_columns) > 10:
                preview += ", ..."

            raise ValueError(
                "The CSV is missing required input columns: "
                + preview
                + ". Use a compatible UNSW-NB15-format CSV."
            )

        # Match training column order.
        model_inputs = features[required_columns]

        # Predict Normal (0) or Attack (1).
        predictions = model.predict(model_inputs)

        # Get estimated attack probabilities, if supported.
        classifier = model.named_steps["classifier"]
        attack_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(model_inputs)
            attack_class_index = list(classifier.classes_).index(1)
            attack_probability = probabilities[:, attack_class_index]

        # Keep original columns and append prediction results.
        output = data.copy()
        output["predicted_label"] = predictions
        output["prediction"] = [
            "Potential Attack" if value == 1 else "Normal"
            for value in predictions
        ]

        if attack_probability is not None:
            output["attack_probability_percent"] = (
                attack_probability * 100
            ).round(2)

        # Prepare a downloadable CSV in memory.
        csv_buffer = BytesIO()
        output.to_csv(
            csv_buffer,
            index=False,
            encoding="utf-8",
        )
        csv_bytes = csv_buffer.getvalue()

        download_token = uuid.uuid4().hex
        download_files[download_token] = csv_bytes
        session["download_token"] = download_token

        # Display only the first 100 rows on the webpage.
        preview_df = output.head(100).astype(object).where(
            pd.notna(output.head(100)), ""
        )

        result = {
            "model_name": MODEL_NAMES[selected_model],
            "total_records": len(output),
            "attack_count": int((predictions == 1).sum()),
            "normal_count": int((predictions == 0).sum()),
            "preview_columns": output.columns.tolist(),
            "preview_rows": preview_df.to_dict(orient="records"),
            "showing": min(len(output), 100),
            "download_url": url_for("download_results"),
        }

        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error=None,
            result=result,
            selected_model=selected_model,
        )

    except (ValueError, pd.errors.EmptyDataError,
            pd.errors.ParserError, UnicodeDecodeError) as exc:
        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error=str(exc),
            result=None,
            selected_model=selected_model,
        )


# --------------------------------------------------
# 6. Download prediction results
# --------------------------------------------------
@app.route("/download")
def download_results():
    token = session.get("download_token")
    csv_bytes = download_files.get(token)

    if csv_bytes is None:
        return (
            "No prediction file is available. Upload a CSV first.",
            404,
        )

    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="nids_predictions.csv",
    )


# --------------------------------------------------
# 7. Start the local development server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
