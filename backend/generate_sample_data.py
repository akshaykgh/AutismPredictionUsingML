from asd_ml.data import ensure_sample_dataset


if __name__ == "__main__":
    destination = ensure_sample_dataset()
    print(f"Sample dataset written to {destination}")
