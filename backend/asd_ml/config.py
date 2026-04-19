from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
GENERATED_DATA_DIR = DATA_DIR / "generated"
MODELS_DIR = ROOT_DIR / "backend" / "models"
REPORTS_DIR = ROOT_DIR / "backend" / "reports"

TARGET_COLUMN = "asd_risk"
AGE_GROUP_COLUMN = "age_group"

SCREENING_COLUMNS = [f"a{i}_score" for i in range(1, 11)]

NUMERIC_COLUMNS = SCREENING_COLUMNS + [
    "age_years",
    "family_asd_score",
    "communication_delay_score",
    "sensory_sensitivity_score",
    "social_responsiveness_score",
]

CATEGORICAL_COLUMNS = [
    AGE_GROUP_COLUMN,
    "sex",
    "ethnicity",
    "jaundice_history",
    "family_asd",
    "used_screening_app_before",
    "speech_delay",
    "anxiety_flag",
    "who_completed_test",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

AGE_GROUPS = [
    "toddler",
    "child",
    "adolescent",
    "adult",
]

AGE_GROUP_RANGES = {
    "toddler": (2, 4),
    "child": (5, 11),
    "adolescent": (12, 17),
    "adult": (18, 45),
}

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
ALL_MODELS_PATH = MODELS_DIR / "all_models.joblib"
ACTIVE_MODEL_PATH = MODELS_DIR / "active_model.json"
