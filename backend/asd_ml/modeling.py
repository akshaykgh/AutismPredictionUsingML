from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_IMPORT_ERROR = None
except Exception as error:  # pragma: no cover - depends on local native runtime
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = str(error)

from .config import (
    ACTIVE_MODEL_PATH,
    AGE_GROUP_COLUMN,
    ALL_MODELS_PATH,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class TrainingArtifacts:
    estimator: GridSearchCV
    metadata: dict


def safe_roc_auc(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return 0.5
    return float(roc_auc_score(y_true, probabilities))


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_search_space(random_state: int = 42) -> list[tuple[str, ImbPipeline, dict]]:
    preprocessor = build_preprocessor()

    model_specs = []

    if XGBClassifier is not None:
        model_specs.append(
            (
                "xgboost",
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=random_state,
                    n_estimators=200,
                ),
                {
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.05, 0.1],
                    "model__subsample": [0.8, 1.0],
                },
            )
        )

    model_specs.extend(
        [
        (
            "random_forest",
            RandomForestClassifier(random_state=random_state),
            {
                "model__n_estimators": [200, 350],
                "model__max_depth": [None, 12],
                "model__min_samples_split": [2, 6],
            },
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(random_state=random_state),
            {
                "model__max_depth": [4, 8, 12],
                "model__min_samples_split": [2, 8, 16],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        ]
    )

    search_space = []
    for name, estimator, grid in model_specs:
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state)),
                ("model", estimator),
            ]
        )
        search_space.append((name, pipeline, grid))
    return search_space


def select_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for threshold in np.arange(0.2, 0.81, 0.02):
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        if precision < 0.45:
            continue
        score = (2.0 * recall + precision) / 3.0
        if score > best_score:
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
            best_score = score
            best_threshold = float(round(threshold, 2))
            best_metrics = {
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1_score(y_true, predictions, zero_division=0),
                "roc_auc": safe_roc_auc(y_true, probabilities),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "true_positives": int(tp),
            }

    return best_threshold, best_metrics


def _feature_importances(pipeline: ImbPipeline) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    ranking = (
        pd.DataFrame({"feature": transformed_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    ranking["importance"] = ranking["importance"].round(4)
    return ranking.to_dict(orient="records")


def train_and_save_models(dataset: pd.DataFrame, random_state: int = 42) -> TrainingArtifacts:
    x = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    best_search: GridSearchCV | None = None
    best_name = ""
    best_roc_auc = float("-inf")

    model_results = []
    trained_bundles: dict[str, dict] = {}
    for name, pipeline, param_grid in build_search_space(random_state=random_state):
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=1,
            refit=True,
        )
        print(pipeline.steps)
        search.fit(x_train, y_train)
        probabilities = search.best_estimator_.predict_proba(x_test)[:, 1]
        threshold, metrics = select_threshold(y_test, probabilities)
        baseline_predictions = (probabilities >= 0.5).astype(int)
        baseline_fn = int(confusion_matrix(y_test, baseline_predictions).ravel()[2])
        optimized_fn = metrics["false_negatives"]
        fn_reduction = round(((baseline_fn - optimized_fn) / baseline_fn) * 100, 2) if baseline_fn else 0.0
        metrics.update(
            {
                "model_name": name,
                "threshold": threshold,
                "best_params": search.best_params_,
                "cv_roc_auc": float(round(search.best_score_, 4)),
                "false_negative_reduction_pct": fn_reduction,
                "feature_importance": _feature_importances(search.best_estimator_),
            }
        )
        model_results.append(metrics)
        trained_bundles[name] = {
            "pipeline": search.best_estimator_,
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
        }

        if best_search is None or metrics["roc_auc"] > best_roc_auc:
            best_search = search
            best_name = name
            best_roc_auc = metrics["roc_auc"]

    if best_search is None:
        raise RuntimeError("Training did not produce a fitted model.")

    best_pipeline = best_search.best_estimator_
    best_probabilities = best_pipeline.predict_proba(x_test)[:, 1]
    best_threshold, best_metrics = select_threshold(y_test, best_probabilities)

    baseline_predictions = (best_probabilities >= 0.5).astype(int)
    baseline_fn = int(confusion_matrix(y_test, baseline_predictions).ravel()[2])
    optimized_fn = best_metrics["false_negatives"]
    fn_reduction = round(((baseline_fn - optimized_fn) / baseline_fn) * 100, 2) if baseline_fn else 0.0

    predictions = (best_probabilities >= best_threshold).astype(int)
    evaluation_frame = x_test.copy()
    evaluation_frame[TARGET_COLUMN] = y_test.values
    evaluation_frame["prediction"] = predictions
    evaluation_frame["probability"] = best_probabilities

    group_metrics = []
    for age_group, group in evaluation_frame.groupby(AGE_GROUP_COLUMN):
        group_metrics.append(
            {
                "age_group": age_group,
                "samples": int(len(group)),
                "accuracy": round(accuracy_score(group[TARGET_COLUMN], group["prediction"]), 4),
                "roc_auc": round(safe_roc_auc(group[TARGET_COLUMN], group["probability"]), 4),
                "positive_rate": round(float(group[TARGET_COLUMN].mean()), 4),
            }
        )

    metadata = {
        "selected_model": best_name,
        "active_model": best_name,
        "available_models": [result["model_name"] for result in model_results],
        "xgboost_enabled": XGBClassifier is not None,
        "xgboost_import_error": XGBOOST_IMPORT_ERROR,
        "decision_threshold": best_threshold,
        "summary_metrics": {
            "accuracy": round(best_metrics["accuracy"], 4),
            "precision": round(best_metrics["precision"], 4),
            "recall": round(best_metrics["recall"], 4),
            "f1": round(best_metrics["f1"], 4),
            "roc_auc": round(best_metrics["roc_auc"], 4),
            "false_negative_reduction_pct": fn_reduction,
        },
        "model_comparison": model_results,
        "age_group_metrics": group_metrics,
        "feature_importance": _feature_importances(best_pipeline),
        "feature_columns": FEATURE_COLUMNS,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "threshold": best_threshold,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )
    joblib.dump(trained_bundles, ALL_MODELS_PATH)
    ACTIVE_MODEL_PATH.write_text(json.dumps({"active": best_name}))
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return TrainingArtifacts(estimator=best_search, metadata=metadata)


def load_trained_model() -> tuple[dict, dict]:
    metadata = json.loads(METADATA_PATH.read_text())
    if ALL_MODELS_PATH.exists() and ACTIVE_MODEL_PATH.exists():
        all_models = joblib.load(ALL_MODELS_PATH)
        active = json.loads(ACTIVE_MODEL_PATH.read_text()).get("active")
        if not active or active not in all_models:
            active = metadata.get("selected_model")
            if active not in all_models:
                active = next(iter(all_models))
            ACTIVE_MODEL_PATH.write_text(json.dumps({"active": active}))
        model_bundle = all_models[active]
        metadata = {**metadata, "active_model": active}
        return model_bundle, metadata

    model_bundle = joblib.load(MODEL_PATH)
    selected = metadata.get("selected_model", "unknown")
    metadata = {**metadata, "active_model": metadata.get("active_model", selected)}
    return model_bundle, metadata


def set_active_model(model_name: str) -> tuple[dict, dict]:
    """Persist which trained classifier is active; requires ``all_models.joblib`` from training."""
    if not ALL_MODELS_PATH.exists():
        raise ValueError(
            "Trained multi-model bundle not found. Run training once (e.g. python backend/train.py or POST /api/retrain) "
            "to enable switching."
        )
    all_models = joblib.load(ALL_MODELS_PATH)
    if model_name not in all_models:
        raise ValueError(f"Unknown model: {model_name!r}")
    metadata = json.loads(METADATA_PATH.read_text())
    available = metadata.get("available_models") or list(all_models.keys())
    if model_name not in available:
        raise ValueError(f"Model {model_name!r} is not available.")
    ACTIVE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODEL_PATH.write_text(json.dumps({"active": model_name}))
    return load_trained_model()
