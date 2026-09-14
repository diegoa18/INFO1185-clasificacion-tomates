from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import discover_samples, validate_sample
from src.protocol import correct_split, load_split, RANDOM_STATE


DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLIT_PATH = ARTIFACTS_DIR / "split.csv"


def archive_legacy_results(results: Path, *, legacy_split: bool = False) -> bool:
    """Detect old outputs even when Git has already delivered the new split.

    Only schema/partition markers are inspected, never historical test scores.
    """
    legacy = legacy_split or (
        results / "classification/bayes_analysis/development_oof_predictions.csv"
    ).exists()
    for relative in ("eda/tomato_features.csv", "segmentation/jaccard_per_image.csv"):
        path = results / relative
        if path.is_file() and "split" not in pd.read_csv(path, nrows=0).columns:
            legacy = True
    features = results / "features/segmented_features.csv"
    if features.is_file():
        partitions = pd.read_csv(features, usecols=["split"])["split"]
        legacy = legacy or "development" in set(partitions)
    if not legacy:
        return False

    archive = results / "legacy_development_test"
    if archive.exists():
        raise FileExistsError(f"Archive already exists; refusing to overwrite: {archive}")
    archive.mkdir(parents=True)
    for name in ("eda", "segmentation", "features", "classification"):
        source = results / name
        if source.exists():
            source.rename(archive / name)
    (archive / "README.md").write_text(
        "# Historical development/test outputs\n\n"
        "Superseded by training/validation/test. These outputs are not valid "
        "for the corrected comparison. Do not use historical test results "
        "for any model-selection decision.\n",
        encoding="utf-8",
    )
    return True


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

    # This versioned artifact is the source of the already-exposed test set.
    existing = pd.read_csv(SPLIT_PATH)
    columns = ["image", "mask", "label"]
    expected = records[columns].sort_values("image").reset_index(drop=True)
    actual = existing[columns].sort_values("image").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Dataset inventory differs from artifacts/split.csv.")
    split = correct_split(existing)

    legacy_split = "development" in set(existing["split"])
    if not legacy_split:
        load_split()
    archive_legacy_results(PROJECT_ROOT / "results", legacy_split=legacy_split)
    if legacy_split:
        split.to_csv(SPLIT_PATH, index=False)

    print("DATASET SPLIT CREATED" if legacy_split else "FROZEN DATASET SPLIT VERIFIED")
    print(f"Total:       {len(split)}")
    print(f"Training:    {(split['split'] == 'training').sum()}")
    print(f"Validation:  {(split['split'] == 'validation').sum()}")
    print(f"Test:        {(split['split'] == 'test').sum()}")
    print(f"Seed:        {RANDOM_STATE}")
    print("Test membership: preserved and fingerprint verified")
    print(f"Output:      {SPLIT_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("CLASS DISTRIBUTION")
    print(split.groupby(["split", "label"]).size())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
