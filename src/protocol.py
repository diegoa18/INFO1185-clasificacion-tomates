"""Image-level holdout protocol; test membership predates this correction."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "split.csv"
RANDOM_STATE = 42
SPLIT_SIZES = {"training": 37, "validation": 12, "test": 13}
CLASS_COUNTS = {
    "training": {"ripe": 18, "unripe": 19},
    "validation": {"ripe": 6, "unripe": 6},
    "test": {"ripe": 7, "unripe": 6},
}
# SHA256 of the sorted original 13 test image paths, joined with '\n'.
# Captured before migration; no image previously in test may enter development.
TEST_MEMBERSHIP_SHA256 = (
    "abaaad163497fc0d45370a38b4f54d3b9a493b8a70cc020fc4a32c4a663eb478"
)


def validate_test_membership(df: pd.DataFrame) -> None:
    images = sorted(df.loc[df["split"] == "test", "image"])
    digest = sha256("\n".join(images).encode()).hexdigest()
    if len(images) != 13 or digest != TEST_MEMBERSHIP_SHA256:
        raise ValueError("Test membership differs from the original frozen 13 images.")


def correct_split(df: pd.DataFrame) -> pd.DataFrame:
    """Subdivide the original non-test pool; independent of input row order."""
    columns = ["image", "mask", "label", "split"]
    if not set(columns).issubset(df.columns):
        raise ValueError("Split requires image, mask, label and split columns.")
    if df[columns].isna().any().any() or df["image"].duplicated().any():
        raise ValueError("Split contains missing values or repeated images.")
    if df["mask"].duplicated().any():
        raise ValueError("Split contains repeated masks.")
    if set(df["split"]) not in (
        {"development", "test"}, set(SPLIT_SIZES)
    ):
        raise ValueError("Unexpected partition names.")
    if df["label"].value_counts().to_dict() != {"ripe": 31, "unripe": 31}:
        raise ValueError("Expected 31 ripe and 31 unripe images.")
    validate_test_membership(df)
    pool = df.loc[df["split"] != "test", columns].sort_values(["label", "image"])
    training, validation = train_test_split(
        pool, test_size=12, stratify=pool["label"], random_state=RANDOM_STATE,
    )
    result = pd.concat([
        training.assign(split="training"),
        validation.assign(split="validation"),
        df.loc[df["split"] == "test", columns],
    ], ignore_index=True)
    return result.sort_values(["split", "label", "image"]).reset_index(drop=True)


def load_split(path: Path = SPLIT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = correct_split(df)
    actual = df.sort_values(["split", "label", "image"]).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Split is not the corrected seed-42 partition. Run create_split.py.")
    for name, counts in CLASS_COUNTS.items():
        if df.loc[df["split"] == name, "label"].value_counts().to_dict() != counts:
            raise ValueError(f"Unexpected class distribution in {name}.")
    return actual


def validate_partition_rows(
    table: pd.DataFrame,
    split: pd.DataFrame,
    partitions: tuple[str, ...],
) -> None:
    """Reject stale feature/result tables, duplicates and mismatched labels."""
    columns = ["image", "label", "split"]
    if not set(columns).issubset(table.columns):
        raise ValueError("Table requires image, label and split columns; regenerate it.")
    actual = table[columns].sort_values("image").reset_index(drop=True)
    expected = split.loc[split["split"].isin(partitions), columns]
    expected = expected.sort_values("image").reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Table does not match the frozen split; regenerate it.")


def split_fingerprint() -> str:
    return sha256(SPLIT_PATH.read_bytes()).hexdigest()
