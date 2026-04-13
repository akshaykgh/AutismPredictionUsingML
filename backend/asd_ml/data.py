from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AGE_GROUP_COLUMN, AGE_GROUP_RANGES, AGE_GROUPS, FEATURE_COLUMNS, GENERATED_DATA_DIR, TARGET_COLUMN


def _build_group_frame(rng: np.random.Generator, age_group: str, size: int) -> pd.DataFrame:
    age_low, age_high = AGE_GROUP_RANGES[age_group]
    screening_base = rng.integers(0, 2, size=(size, 10))
    social_signal = rng.normal(loc=0.0, scale=1.0, size=size)

    frame = pd.DataFrame(
        {
            AGE_GROUP_COLUMN: age_group,
            "age_years": rng.integers(age_low, age_high + 1, size=size),
            "sex": rng.choice(["male", "female"], size=size, p=[0.62, 0.38]),
            "ethnicity": rng.choice(
                ["asian", "black", "hispanic", "middle eastern", "white", "other"],
                size=size,
                p=[0.19, 0.11, 0.17, 0.06, 0.39, 0.08],
            ),
            "jaundice_history": rng.choice(["yes", "no"], size=size, p=[0.14, 0.86]),
            "family_asd": rng.choice(["yes", "no"], size=size, p=[0.22, 0.78]),
            "used_screening_app_before": rng.choice(["yes", "no"], size=size, p=[0.31, 0.69]),
            "speech_delay": rng.choice(["yes", "no"], size=size, p=[0.21, 0.79]),
            "anxiety_flag": rng.choice(["yes", "no"], size=size, p=[0.29, 0.71]),
            "who_completed_test": rng.choice(
                ["self", "parent", "caregiver", "clinician"],
                size=size,
                p=[0.42, 0.28, 0.12, 0.18],
            ),
            "family_asd_score": rng.normal(loc=0.35, scale=0.18, size=size).clip(0, 1),
            "communication_delay_score": rng.normal(loc=0.42, scale=0.22, size=size).clip(0, 1),
            "sensory_sensitivity_score": rng.normal(loc=0.40, scale=0.21, size=size).clip(0, 1),
            "social_responsiveness_score": rng.normal(loc=0.48, scale=0.20, size=size).clip(0, 1),
        }
    )

    for index in range(10):
        frame[f"a{index + 1}_score"] = screening_base[:, index]

    screening_sum = frame[[f"a{i}_score" for i in range(1, 11)]].sum(axis=1)
    logit = (
        -7.0
        + 0.62 * screening_sum
        + 2.0 * frame["communication_delay_score"]
        + 1.5 * frame["sensory_sensitivity_score"]
        + 1.2 * frame["social_responsiveness_score"]
        + 1.3 * (frame["family_asd"] == "yes").astype(float)
        + 0.7 * (frame["speech_delay"] == "yes").astype(float)
        + 0.4 * (frame["jaundice_history"] == "yes").astype(float)
        + 0.5 * social_signal
    )

    group_adjustment = {
        "toddler": 0.45,
        "child": 0.25,
        "adolescent": 0.1,
        "adult": -0.05,
    }[age_group]
    probabilities = 1 / (1 + np.exp(-(logit + group_adjustment)))
    frame[TARGET_COLUMN] = rng.binomial(1, probabilities).astype(int)
    return frame


def generate_synthetic_dataset(rows_per_group: int = 450, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [_build_group_frame(rng, age_group, rows_per_group) for age_group in AGE_GROUPS]
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return dataset


def ensure_sample_dataset(output_path: Path | None = None, rows_per_group: int = 450, seed: int = 42) -> Path:
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = output_path or GENERATED_DATA_DIR / "asd_screening_sample.csv"
    if not destination.exists():
        dataset = generate_synthetic_dataset(rows_per_group=rows_per_group, seed=seed)
        dataset.to_csv(destination, index=False)
    return destination


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    if dataset_path is None:
        dataset_path = ensure_sample_dataset()
    frame = pd.read_csv(dataset_path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame
