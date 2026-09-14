from __future__ import annotations
import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from src.features import load_mask, load_rgb_image
from src.protocol import load_split, split_fingerprint
from src.segmentation import (
    CHANNEL_COMBINATIONS,
    KMEANS_INIT,
    KMEANS_N_CLUSTERS,
    KMEANS_PARAMETERS,
    KMEANS_THREADS,
    SELECTION_PATH,
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


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        results
        .groupby(
            ["split", "channels"],
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
        .sort_values(["split", "order"])
        .drop(columns="order")
        .reset_index(drop=True)
    )

    return summary


def main() -> int:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}"
        )

    split = load_split()

    development = split[
        split["split"].isin(["training", "validation"])
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
                **KMEANS_PARAMETERS,
            )

            jaccard = jaccard_index(
                predicted=result.mask,
                reference=reference_mask,
            )

            rows.append(
                {
                    "image": sample.image,
                    "label": sample.label,
                    "split": sample.split,
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

    # idxmax chooses the first maximum in the documented channel order.
    validation_summary = summary.loc[summary["split"] == "validation"]
    best = validation_summary.loc[validation_summary["mean_jaccard"].idxmax()]
    selection = {
        "channels": str(best["channels"]),
        "selection_partition": "validation",
        "criterion": "maximum mean Jaccard; ties in R,G,B,RG,RB,GB,RGB order",
        "mean_jaccard": float(best["mean_jaccard"]),
        "n_clusters": KMEANS_N_CLUSTERS,
        "init": KMEANS_INIT,
        "threads": KMEANS_THREADS,
        "parameters": KMEANS_PARAMETERS,
        "split_sha256": split_fingerprint(),
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Selected on validation: {best['channels']}")

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
        f"{KMEANS_N_CLUSTERS}"
    )
    print(
        "Initialization:  "
        f"{KMEANS_INIT}"
    )
    print(
        f"n_init:          "
        f"{KMEANS_PARAMETERS['n_init']}"
    )
    print(
        f"max_iter:        "
        f"{KMEANS_PARAMETERS['max_iter']}"
    )
    print(
        f"tol:             "
        f"{KMEANS_PARAMETERS['tol']}"
    )
    print(
        f"Seed:            "
        f"{KMEANS_PARAMETERS['random_state']}"
    )
    print(
        f"Border fraction: "
        f"{KMEANS_PARAMETERS['border_fraction']}"
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
