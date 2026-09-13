from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.features import load_mask, load_rgb_image
from src.segmentation import (
    CHANNEL_COMBINATIONS,
    jaccard_index,
    segment_kmeans,
)


SPLIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "split.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
)
PER_IMAGE_PATH = (
    OUTPUT_DIR
    / "jaccard_per_image.csv"
)
SUMMARY_PATH = (
    OUTPUT_DIR
    / "jaccard_summary.csv"
)
RANDOM_STATE = 42
N_CLUSTERS = 2
N_INIT = 10
MAX_ITER = 300
TOL = 1e-4
BORDER_FRACTION = 0.05


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        results
        .groupby(
            "channels",
            as_index=False,
        )
        .agg(
            mean_jaccard=(
                "jaccard",
                "mean",
            ),
            std_jaccard=(
                "jaccard",
                "std",
            ),
            median_jaccard=(
                "jaccard",
                "median",
            ),
            min_jaccard=(
                "jaccard",
                "min",
            ),
            max_jaccard=(
                "jaccard",
                "max",
            ),
        )
    )

    order = {
        channels: index
        for index, channels
        in enumerate(
            CHANNEL_COMBINATIONS
        )
    }

    summary["order"] = (
        summary["channels"]
        .map(order)
    )

    summary = (
        summary
        .sort_values("order")
        .drop(columns="order")
        .reset_index(drop=True)
    )

    return summary


def main() -> int:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = pd.read_csv(
        SPLIT_PATH
    )

    development = split[
        split["split"] == "development"
    ].copy()

    if development.empty:
        raise RuntimeError(
            "No development samples found."
        )

    rows: list[
        dict[str, object]
    ] = []

    total_runs = (
        len(development)
        * len(CHANNEL_COMBINATIONS)
    )

    current_run = 0

    for sample in development.itertuples(
        index=False
    ):
        image_path = (
            PROJECT_ROOT
            / sample.image
        )

        mask_path = (
            PROJECT_ROOT
            / sample.mask
        )

        rgb = load_rgb_image(
            image_path
        )

        reference_mask = load_mask(
            mask_path
        )

        if (
            rgb.shape[:2]
            != reference_mask.shape
        ):
            raise ValueError(
                "Image/mask shape mismatch for "
                f"{sample.image}"
            )

        for channels in CHANNEL_COMBINATIONS:
            current_run += 1

            print(
                f"[{current_run:03d}/"
                f"{total_runs:03d}] "
                f"{Path(sample.image).name} "
                f"- {channels}"
            )

            result = segment_kmeans(
                rgb=rgb,
                channels=channels,
                random_state=RANDOM_STATE,
                n_init=N_INIT,
                max_iter=MAX_ITER,
                tol=TOL,
                border_fraction=BORDER_FRACTION,
            )

            jaccard = jaccard_index(
                predicted=result.mask,
                reference=reference_mask,
            )

            rows.append(
                {
                    "image": sample.image,
                    "label": sample.label,
                    "channels": channels,
                    "jaccard": jaccard,
                    "tomato_cluster": (
                        result.tomato_cluster
                    ),
                    "border_occupancy_cluster_0": (
                        result
                        .border_occupancy_cluster_0
                    ),
                    "border_occupancy_cluster_1": (
                        result
                        .border_occupancy_cluster_1
                    ),
                    "inertia": (
                        result.inertia
                    ),
                    "n_iter": (
                        result.n_iter
                    ),
                }
            )

    results = pd.DataFrame(
        rows
    )

    expected_rows = total_runs

    if len(results) != expected_rows:
        raise RuntimeError(
            f"Unexpected result count: "
            f"{len(results)} "
            f"(expected {expected_rows})"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        PER_IMAGE_PATH,
        index=False,
    )

    summary = create_summary(
        results
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print(
        "K-MEANS SEGMENTATION COMPLETE"
    )
    print("=" * 60)

    print(
        f"Images:       "
        f"{len(development)}"
    )
    print(
        f"Combinations: "
        f"{len(CHANNEL_COMBINATIONS)}"
    )
    print(
        f"Runs:         "
        f"{len(results)}"
    )

    print()
    print("PARAMETERS")
    print("-" * 60)

    print(
        f"Clusters:        "
        f"{N_CLUSTERS}"
    )
    print(
        "Initialization:  "
        "k-means++"
    )
    print(
        f"n_init:          "
        f"{N_INIT}"
    )
    print(
        f"max_iter:        "
        f"{MAX_ITER}"
    )
    print(
        f"tol:             "
        f"{TOL}"
    )
    print(
        f"Seed:            "
        f"{RANDOM_STATE}"
    )
    print(
        f"Border fraction: "
        f"{BORDER_FRACTION}"
    )

    print()
    print("JACCARD SUMMARY")
    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print()
    print("OUTPUT")
    print(
        PER_IMAGE_PATH
        .relative_to(PROJECT_ROOT)
    )

    print(
        SUMMARY_PATH
        .relative_to(PROJECT_ROOT)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
