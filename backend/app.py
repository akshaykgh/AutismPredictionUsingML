from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from asd_ml.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
from asd_ml.data import ensure_sample_dataset
from asd_ml.modeling import load_trained_model, train_and_save_models


def bootstrap_model() -> tuple[dict, dict]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        dataset = pd.read_csv(ensure_sample_dataset())
        train_and_save_models(dataset)
    return load_trained_model()


model_bundle, model_metadata = bootstrap_model()

app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/api/metadata")
def metadata():
    return jsonify(model_metadata)


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400

    input_frame = pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}])
    print(model_bundle)
    probability = float(model_bundle["pipeline"].predict_proba(input_frame)[:, 1][0])
    threshold = float(model_bundle["threshold"])
    prediction = int(probability >= threshold)

    response = {
        "probability": round(probability, 4),
        "threshold": threshold,
        "prediction": prediction,
        "label": "High ASD risk" if prediction else "Low ASD risk",
    }
    return jsonify(response)


@app.post("/api/retrain")
def retrain():
    global model_bundle, model_metadata
    payload = request.get_json(silent=True) or {}
    dataset_path = payload.get("dataset_path")
    dataset = pd.read_csv(Path(dataset_path)) if dataset_path else pd.read_csv(ensure_sample_dataset())
    artifacts = train_and_save_models(dataset)
    model_bundle, model_metadata = load_trained_model()
    return jsonify(
        {
            "status": "retrained",
            "selected_model": artifacts.metadata["selected_model"],
            "summary_metrics": artifacts.metadata["summary_metrics"],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
