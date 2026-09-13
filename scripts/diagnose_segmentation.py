from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.features import load_mask, load_rgb_image
from src.segmentation import (
    CHANNEL_COMBINATIONS,
    jaccard_index,
    segment_kmeans,
)


SPLIT_PATH = PROJECT_ROOT / "artifacts" / "split.csv"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
)
DETAIL_PATH = (
    OUTPUT_DIR
    / "cluster_selection_diagnostic.csv"
)
SUMMARY_PATH = (
    OUTPUT_DIR
    / "cluster_selection_summary.csv"
)
RANDOM_STATE = 42
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
            mean_selected_jaccard=(
                "selected_jaccard",
                "mean",
            ),
            mean_oracle_jaccard=(
                "oracle_jaccard",
                "mean",
            ),
            median_selected_jaccard=(
                "selected_jaccard",
                "median",
            ),
            median_oracle_jaccard=(
                "oracle_jaccard",
                "median",
            ),
            mean_selection_loss=(
                "selection_loss",
                "mean",
            ),
            cluster_selection_accuracy=(
                "selection_correct",
                "mean",
            ),
        )
    )

    order = {
        channels: index
        for index, channels
        in enumerate(CHANNEL_COMBINATIONS)
    }

    summary["order"] = (
        summary["channels"]
        .map(order)
    )

    return (
        summary
        .sort_values("order")
        .drop(columns="order")
        .reset_index(drop=True)
    )


def main() -> int:
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

    run = 0

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

        reference = load_mask(
            mask_path
        )

        for channels in CHANNEL_COMBINATIONS:
            run += 1

            print(
                f"[{run:03d}/{total_runs:03d}] "
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

            selected_mask = result.mask
            alternative_mask = ~selected_mask

            selected_jaccard = (
                jaccard_index(
                    selected_mask,
                    reference,
                )
            )

            alternative_jaccard = (
                jaccard_index(
                    alternative_mask,
                    reference,
                )
            )

            oracle_jaccard = max(
                selected_jaccard,
                alternative_jaccard,
            )

            selection_loss = (
                oracle_jaccard
                - selected_jaccard
            )

            selection_correct = (
                selected_jaccard
                >= alternative_jaccard
            )

            rows.append(
                {
                    "image": sample.image,
                    "label": sample.label,
                    "channels": channels,
                    "selected_jaccard": (
                        selected_jaccard
                    ),
                    "alternative_jaccard": (
                        alternative_jaccard
                    ),
                    "oracle_jaccard": (
                        oracle_jaccard
                    ),
                    "selection_loss": (
                        selection_loss
                    ),
                    "selection_correct": (
                        selection_correct
                    ),
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
                }
            )

    results = pd.DataFrame(
        rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAIL_PATH,
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
        "CLUSTER SELECTION DIAGNOSTIC"
    )
    print(
        summary.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print()
    print("INTERPRETATION")
    print(
        "selected = Jaccard obtained using "
        "the automatic border heuristic"
    )
    print(
        "oracle   = best Jaccard obtainable "
        "from either K-Means cluster"
    )
    print(
        "loss     = oracle - selected"
    )
    print(
        "accuracy = fraction of images where "
        "the heuristic chose the better cluster"
    )

    print()
    print("OUTPUT")
    print("-" * 80)

    print(
        DETAIL_PATH
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
