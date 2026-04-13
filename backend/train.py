from __future__ import annotations

import argparse
import json

from asd_ml.data import ensure_sample_dataset, load_dataset
from asd_ml.modeling import train_and_save_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASD screening models and save artifacts.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a CSV dataset. If omitted, a synthetic sample dataset is generated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset or ensure_sample_dataset()
    dataset = load_dataset(dataset_path)
    artifacts = train_and_save_models(dataset)
    print(json.dumps(artifacts.metadata, indent=2))
