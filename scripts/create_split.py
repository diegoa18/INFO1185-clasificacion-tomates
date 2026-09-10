from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import discover_samples, validate_sample


DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLIT_PATH = ARTIFACTS_DIR / "split.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

def main() -> int:
    samples = discover_samples(DATASET_DIR)
    errors: list[str] = []
    for sample in samples:
        errors.extend(validate_sample(sample))

    if errors:
        print("ERROR: dataset validation failed.")
        for error in errors:
            print(f"- {error}")
        return 1

    records = pd.DataFrame(
        {
            "image": [
                str(sample.image_path.relative_to(PROJECT_ROOT))
                for sample in samples
            ],
            "mask": [
                str(sample.mask_path.relative_to(PROJECT_ROOT))
                for sample in samples
            ],
            "label": [sample.label for sample in samples],
        }
    )

    development, test = train_test_split(records, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=records["label"],)
    development = development.assign(split="development")
    test = test.assign(split="test")

    split = (
        pd.concat([development, test], ignore_index=True)
        .sort_values(["split", "label", "image"])
        .reset_index(drop=True)
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)

    print("DATASET SPLIT CREATED")
    print(f"Total:       {len(split)}")
    print(f"Development: {(split['split'] == 'development').sum()}")
    print(f"Test:        {(split['split'] == 'test').sum()}")
    print(f"Seed:        {RANDOM_STATE}")
    print(f"Test size:   {TEST_SIZE}")
    print(f"Output:      {SPLIT_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("CLASS DISTRIBUTION")
    print(split.groupby(["split", "label"]).size())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
