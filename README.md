# Autism Prediction Using ML

Full-stack ASD screening project with:

- `XGBoost`, `Random Forest`, and `Decision Tree` classifiers
- `SMOTE` for class imbalance handling
- `GridSearchCV` for hyperparameter tuning
- threshold optimization to reduce false negatives
- a `Flask` API for training metadata and real-time predictions
- a `React` dashboard for interactive screening and model explainability

## Project Structure

```text
backend/
  app.py                  Flask API
  train.py                Model training entrypoint
  generate_sample_data.py Synthetic dataset generator
  requirements.txt        Python dependencies
  asd_ml/
    config.py             Shared schema and paths
    data.py               Dataset loading and synthetic data generation
    modeling.py           Preprocessing, training, evaluation, persistence
frontend/
  package.json
  src/
    App.jsx               Dashboard UI
    styles.css            Custom dashboard styling
data/
  generated/              Auto-created sample dataset output
```

## Dataset Schema

The backend is designed to work with a CSV that contains:

- categorical fields: `age_group`, `sex`, `ethnicity`, `jaundice_history`, `family_asd`, `used_screening_app_before`, `speech_delay`, `anxiety_flag`, `who_completed_test`
- numeric fields: `age_years`, `a1_score` through `a10_score`, `family_asd_score`, `communication_delay_score`, `sensory_sensitivity_score`, `social_responsiveness_score`
- target field: `asd_risk`

If you do not provide a dataset, the project generates a synthetic ASD-style screening dataset across four age groups: `toddler`, `child`, `adolescent`, and `adult`.

## Backend Setup

```bash
cd /Users/akshaykumargh/Desktop/AutismPredictionUsingML
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r backend/requirements.txt
python3 backend/train.py
python3 backend/app.py
```

Optional training with your own dataset:

```bash
python3 backend/train.py --dataset /absolute/path/to/asd_dataset.csv
```

Available API routes:

- `GET /api/health`
- `GET /api/metadata`
- `POST /api/predict`
- `POST /api/retrain`

Example prediction payload:

```json
{
  "age_group": "child",
  "age_years": 8,
  "sex": "male",
  "ethnicity": "white",
  "jaundice_history": "no",
  "family_asd": "yes",
  "used_screening_app_before": "no",
  "speech_delay": "yes",
  "anxiety_flag": "no",
  "who_completed_test": "parent",
  "family_asd_score": 0.65,
  "communication_delay_score": 0.72,
  "sensory_sensitivity_score": 0.58,
  "social_responsiveness_score": 0.67,
  "a1_score": 1,
  "a2_score": 1,
  "a3_score": 0,
  "a4_score": 1,
  "a5_score": 1,
  "a6_score": 0,
  "a7_score": 1,
  "a8_score": 1,
  "a9_score": 0,
  "a10_score": 1
}
```

## Frontend Setup

In another terminal:

```bash
cd /Users/akshaykumargh/Desktop/AutismPredictionUsingML/frontend
npm install
npm run dev
```

The dashboard expects the Flask API at `http://localhost:5000/api`.

To override that:

```bash
echo 'VITE_API_BASE=http://localhost:5000/api' > frontend/.env
```

## Notes

- The saved model artifacts are written to `backend/models/`.
- Training metadata, model comparison, age-group metrics, and feature importances are exposed through `/api/metadata`.
- The current implementation uses a synthetic fallback dataset because no raw ASD CSV was present in the repository.
