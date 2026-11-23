
from pathlib import Path
from io import BytesIO
import uuid
import os

from dotenv import load_dotenv
load_dotenv()

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

from huggingface_hub import hf_hub_download


# --------------------------------------------------
# 1. Initialize Flask
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "local-development-secret-change-before-deployment"
)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# --------------------------------------------------
# 2. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"


# --------------------------------------------------
# 3. Hugging Face configuration
# --------------------------------------------------

MODEL_REPO = os.environ.get("HF_MODEL_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")

if not MODEL_REPO:
    raise RuntimeError("HF_MODEL_REPO environment variable is missing.")

print("Downloading GA-optimized NIDS model...")

OPTIMIZED_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="optimized_rf.joblib",
    token=HF_TOKEN,
)

optimized_model = joblib.load(OPTIMIZED_PATH)

print("GA-optimized model loaded successfully!")


# --------------------------------------------------
# 4. Model configuration
# --------------------------------------------------

MODEL_NAME = "GA-Optimized Random Forest"


# --------------------------------------------------
# 5. Load comparison results
# --------------------------------------------------

if not COMPARISON_PATH.exists():
    raise FileNotFoundError(
        f"Comparison file not found: {COMPARISON_PATH}"
    )

comparison_df = pd.read_csv(COMPARISON_PATH)


# --------------------------------------------------
# 6. Temporary download storage
# --------------------------------------------------

download_files = {}


# --------------------------------------------------
# 7. Dashboard
# --------------------------------------------------

@app.route("/")
def dashboard():

    comparison = comparison_df.to_dict(orient="records")

    return render_template(
        "index.html",
        comparison=comparison,
        error=None,
        result=None,
    )


# --------------------------------------------------
# 8. Prediction
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    uploaded_file = request.files.get("file")

    if not uploaded_file or uploaded_file.filename == "":
        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error="Please choose a CSV file first.",
            result=None,
        )

    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error="Only CSV files are supported.",
            result=None,
        )

    try:

        data = pd.read_csv(uploaded_file)

        data.columns = data.columns.astype(str).str.strip()

        if data.empty:
            raise ValueError("The uploaded CSV contains no records.")

        if data.columns.duplicated().any():
            raise ValueError("The CSV contains duplicate column names.")

        features = data.drop(
            columns=["label", "attack_cat", "id"],
            errors="ignore",
        )

        required_columns = list(
            optimized_model.named_steps[
                "preprocessor"
            ].feature_names_in_
        )

        missing_columns = [
            column for column in required_columns
            if column not in features.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing_columns[:10])
                + ". Upload a compatible UNSW-NB15 CSV."
            )

        model_inputs = features[required_columns]

        predictions = optimized_model.predict(model_inputs)

        probabilities = optimized_model.predict_proba(model_inputs)

        classifier = optimized_model.named_steps["classifier"]

        attack_class_index = list(classifier.classes_).index(1)

        attack_probability = probabilities[:, attack_class_index]

        output = data.copy()

        output["predicted_label"] = predictions

        output["prediction"] = [
            "Potential Attack" if value == 1 else "Normal"
            for value in predictions
        ]

        output["attack_probability_percent"] = (
            attack_probability * 100
        ).round(2)

        csv_buffer = BytesIO()

        output.to_csv(csv_buffer, index=False, encoding="utf-8")

        download_token = uuid.uuid4().hex

        download_files[download_token] = csv_buffer.getvalue()

        session["download_token"] = download_token

        preview_df = output.head(100).astype(object).where(
            pd.notna(output.head(100)), ""
        )

        result = {
            "model_name": MODEL_NAME,
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
        )

    except (
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:

        return render_template(
            "index.html",
            comparison=comparison_df.to_dict(orient="records"),
            error=str(exc),
            result=None,
        )


# --------------------------------------------------
# 9. Download predictions
# --------------------------------------------------

@app.route("/download")
def download_results():

    token = session.get("download_token")

    csv_bytes = download_files.get(token)

    if csv_bytes is None:
        return "No prediction file available. Upload a CSV first.", 404

    return send_file(
        BytesIO(csv_bytes),
        mimetype="text/csv",
        as_attachment=True,
        download_name="nids_predictions.csv",
    )


# --------------------------------------------------
# 10. Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
    )
