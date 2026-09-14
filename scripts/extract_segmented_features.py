from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.features import (
    extract_mean_color_features,
    load_mask,
    load_rgb_image,
)
from src.segmentation import (
    KMEANS_PARAMETERS,
    jaccard_index,
    segment_kmeans,
    load_selected_channels,
)
from src.protocol import load_split


SPLIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "split.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "features"
)
OUTPUT_PATH = (
    OUTPUT_DIR
    / "segmented_features.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frozen segmentation features.")
    parser.add_argument("--include-test", action="store_true",
                        help="Include test inference for the final frozen evaluation.")
    args = parser.parse_args()
    channels = load_selected_channels()
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = load_split()
    if not args.include_test:
        split = split.loc[split["split"] != "test"].copy()

    required_columns = {
        "image",
        "mask",
        "label",
        "split",
    }

    missing_columns = (
        required_columns
        - set(split.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing split columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    rows: list[
        dict[str, object]
    ] = []

    total = len(split)

    for index, sample in enumerate(
        split.itertuples(index=False),
        start=1,
    ):
        image_path = (
            PROJECT_ROOT
            / sample.image
        )

        mask_path = (
            PROJECT_ROOT
            / sample.mask
        )

        print(
            f"[{index:02d}/{total:02d}] "
            f"{Path(sample.image).name} "
            f"({sample.split})"
        )

        rgb = load_rgb_image(
            image_path
        )

        reference_mask = load_mask(
            mask_path
        )

        segmentation = segment_kmeans(
            rgb=rgb,
            channels=channels,
            **KMEANS_PARAMETERS,
        )

        predicted_mask = (
            segmentation.mask
        )

        features = (
            extract_mean_color_features(
                rgb=rgb,
                mask=predicted_mask,
            )
        )

        jaccard = jaccard_index(
            predicted=predicted_mask,
            reference=reference_mask,
        )

        foreground_pixels = int(
            predicted_mask.sum()
        )

        total_pixels = int(
            predicted_mask.size
        )

        foreground_fraction = (
            foreground_pixels
            / total_pixels
        )

        rows.append(
            {
                "image": sample.image,
                "label": sample.label,
                "split": sample.split,
                "segmentation_channels": (
                    channels
                ),
                "segmentation_jaccard": (
                    jaccard
                ),
                "foreground_pixels": (
                    foreground_pixels
                ),
                "foreground_fraction": (
                    foreground_fraction
                ),
                **features,
            }
        )

    output = pd.DataFrame(
        rows
    )

    if len(output) != len(split):
        raise RuntimeError(
            "Unexpected number of extracted samples"
        )

    feature_columns = [
        "R",
        "G",
        "B",
        "H",
        "S",
        "V",
    ]

    if (
        output[feature_columns]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Extracted features contain NaN values"
        )

    if not np.isfinite(
        output[feature_columns]
        .to_numpy()
    ).all():
        raise RuntimeError(
            "Extracted features contain non-finite values"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "SEGMENTED FEATURES EXTRACTED"
    )
    print(
        f"Samples:      {len(output)}"
    )

    print(
        f"Training:     "
        f"{(output['split'] == 'training').sum()}"
    )
    print(f"Validation:   {(output['split'] == 'validation').sum()}")

    print(
        f"Test:         "
        f"{(output['split'] == 'test').sum()}"
    )

    print(
        f"Channels:     "
        f"{channels}"
    )

    print()
    print("FEATURES - TRAINING/VALIDATION ONLY")
    print(
        output.loc[output["split"] != "test",
            feature_columns
        ]
        .describe()
        .T
        .to_string(
            float_format=(
                lambda value: f"{value:.4f}"
            )
        )
    )

    print()
    print("SEGMENTATION QUALITY - TRAINING/VALIDATION ONLY")
    print(
        output.loc[output["split"] != "test"].groupby(
            "split"
        )["segmentation_jaccard"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
        .to_string(
            float_format=(
                lambda value: f"{value:.4f}"
            )
        )
    )

    print()
    print("OUTPUT")
    print(
        OUTPUT_PATH
        .relative_to(PROJECT_ROOT)
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
