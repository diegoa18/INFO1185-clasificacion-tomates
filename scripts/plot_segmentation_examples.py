from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import load_mask, load_rgb_image
from src.segmentation import jaccard_index, segment_kmeans


RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "jaccard_per_image.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "figures"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "segmentation"
    / "representative_examples.csv"
)
CHANNELS = "RG"
RANDOM_STATE = 42
N_INIT = 10
MAX_ITER = 300
TOL = 1e-4
BORDER_FRACTION = 0.05


def select_representative_examples(
    results: pd.DataFrame,
) -> pd.DataFrame:
    channel_results = (
        results[
            results["channels"] == CHANNELS
        ]
        .copy()
        .sort_values("jaccard")
        .reset_index(drop=True)
    )

    if channel_results.empty:
        raise RuntimeError(
            f"No results found for channels {CHANNELS}."
        )

    worst = channel_results.iloc[0]
    median_value = channel_results["jaccard"].median()
    median_index = (
        channel_results["jaccard"]
        .sub(median_value)
        .abs()
        .idxmin()
    )

    median = channel_results.loc[median_index]
    best = channel_results.iloc[-1]
    examples = pd.DataFrame(
        [
            {
                **worst.to_dict(),
                "example_type": "worst",
            },
            {
                **median.to_dict(),
                "example_type": "median",
            },
            {
                **best.to_dict(),
                "example_type": "best",
            },
        ]
    )

    return examples


def plot_single_example(
    image_path: Path,
    mask_path: Path,
    expected_jaccard: float,
    example_type: str,
    label: str,
    output_path: Path,
) -> float:
    rgb = load_rgb_image(
        image_path
    )

    reference = load_mask(
        mask_path
    )

    result = segment_kmeans(
        rgb=rgb,
        channels=CHANNELS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
        tol=TOL,
        border_fraction=BORDER_FRACTION,
    )

    reproduced_jaccard = jaccard_index(
        predicted=result.mask,
        reference=reference,
    )

    if abs(
        reproduced_jaccard
        - expected_jaccard
    ) > 1e-9:
        raise RuntimeError(
            "Reproduced Jaccard does not match stored result: "
            f"stored={expected_jaccard:.10f}, "
            f"reproduced={reproduced_jaccard:.10f}"
        )

    class_name = (
        "Maduro"
        if label == "ripe"
        else "Inmaduro"
    )

    example_name = {
        "best": "Mejor caso",
        "median": "Caso mediano",
        "worst": "Peor caso",
    }[example_type]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
    )

    axes[0].imshow(rgb)
    axes[0].set_title("Imagen original")
    axes[1].imshow(
        reference,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Máscara de referencia")
    axes[2].imshow(
        result.mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"K-Means {CHANNELS}"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{example_name} - {class_name} - "
        f"Jaccard = {reproduced_jaccard:.4f}"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return reproduced_jaccard


def plot_combined_examples(
    examples: pd.DataFrame,
    reproduced: dict[str, tuple],
    output_path: Path,
) -> None:
    order = [
        "best",
        "median",
        "worst",
    ]

    titles = {
        "best": "Mejor caso",
        "median": "Caso mediano",
        "worst": "Peor caso",
    }

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12),
    )

    for row_index, example_type in enumerate(order):
        row = examples[
            examples["example_type"]
            == example_type
        ].iloc[0]

        rgb, reference, predicted, jaccard = (
            reproduced[example_type]
        )

        axes[row_index, 0].imshow(rgb)
        axes[row_index, 1].imshow(
            reference,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 2].imshow(
            predicted,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 0].set_ylabel(
            f"{titles[example_type]}\n"
            f"J = {jaccard:.4f}",
            fontsize=11,
        )

        class_name = (
            "Maduro"
            if row["label"] == "ripe"
            else "Inmaduro"
        )

        axes[row_index, 0].set_title(
            f"Original\n{class_name}"
        )

        axes[row_index, 1].set_title(
            "Referencia"
        )

        axes[row_index, 2].set_title(
            f"K-Means {CHANNELS}"
        )

        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle(
        "Ejemplos representativos de segmentación con K-Means RG",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> int:
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing segmentation results: {RESULTS_PATH}"
        )

    results = pd.read_csv(
        RESULTS_PATH
    )

    required_columns = {
        "image",
        "label",
        "channels",
        "jaccard",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    examples = select_representative_examples(
        results
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reproduced: dict[
        str,
        tuple,
    ] = {}

    output_rows: list[
        dict[str, object]
    ] = []

    for row in examples.itertuples(
        index=False
    ):
        image_path = (
            PROJECT_ROOT
            / row.image
        )

        image_relative = Path(
            row.image
        )

        mask_path = (
            PROJECT_ROOT
            / image_relative.parent.parent
            / "masks"
            / f"{image_relative.stem}_mask.png"
        )

        output_path = (
            OUTPUT_DIR
            / f"rg_{row.example_type}.png"
        )

        reproduced_jaccard = (
            plot_single_example(
                image_path=image_path,
                mask_path=mask_path,
                expected_jaccard=float(
                    row.jaccard
                ),
                example_type=row.example_type,
                label=row.label,
                output_path=output_path,
            )
        )

        rgb = load_rgb_image(
            image_path
        )

        reference = load_mask(
            mask_path
        )

        segmentation = segment_kmeans(
            rgb=rgb,
            channels=CHANNELS,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            tol=TOL,
            border_fraction=BORDER_FRACTION,
        )

        reproduced[
            row.example_type
        ] = (
            rgb,
            reference,
            segmentation.mask,
            reproduced_jaccard,
        )

        output_rows.append(
            {
                "example_type": (
                    row.example_type
                ),
                "image": row.image,
                "label": row.label,
                "channels": CHANNELS,
                "jaccard": (
                    reproduced_jaccard
                ),
            }
        )

    plot_combined_examples(
        examples=pd.DataFrame(
            output_rows
        ),
        reproduced=reproduced,
        output_path=(
            OUTPUT_DIR
            / "rg_representative_examples.png"
        ),
    )

    output_summary = pd.DataFrame(
        output_rows
    )

    output_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "SEGMENTATION EXAMPLES GENERATED"
    )
    print("=" * 60)

    print(
        output_summary.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.4f}"
            ),
        )
    )

    print()
    print("OUTPUT")
    print(
        SUMMARY_PATH
        .relative_to(PROJECT_ROOT)
    )

    print(
        OUTPUT_DIR
        .relative_to(PROJECT_ROOT)
        / "rg_best.png"
    )

    print(
        OUTPUT_DIR
        .relative_to(PROJECT_ROOT)
        / "rg_median.png"
    )

    print(
        OUTPUT_DIR
        .relative_to(PROJECT_ROOT)
        / "rg_worst.png"
    )

    print(
        OUTPUT_DIR
        .relative_to(PROJECT_ROOT)
        / "rg_representative_examples.png"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
