from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import extract_color_statistics
from src.protocol import load_split


SPLIT_PATH = PROJECT_ROOT / "artifacts" / "split.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "eda" / "tomato_features.csv"


def main() -> int:
    split = load_split()
    training = split[split["split"] == "training"].copy()

    if training.empty:
        raise RuntimeError("No training samples found.")

    rows: list[dict[str, object]] = []

    for row in training.itertuples(index=False):
        image_path = PROJECT_ROOT / row.image
        mask_path = PROJECT_ROOT / row.mask

        statistics = extract_color_statistics(
            image_path=image_path,
            mask_path=mask_path,
        )

        rows.append(
            {
                "image": row.image,
                "mask": row.mask,
                "label": row.label,
                "split": row.split,
                **statistics,
            }
        )

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("EDA FEATURES EXTRACTED")
    print(f"Samples: {len(output)}")
    print(f"Features: {len(output.columns) - 4}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("Class distribution")
    print(output["label"].value_counts().sort_index())

    print()
    print("Preview")
    print(output[["image", "label", "fruit_R_mean", "fruit_G_mean",
                   "fruit_B_mean", "fruit_H_mean", "fruit_S_mean",
                   "fruit_V_mean"]].head().to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
