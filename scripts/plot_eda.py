from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "eda" / "tomato_features.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "eda" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "eda" / "tables"


def save_region_boxplot(
    df: pd.DataFrame,
    features: list[str],
    title: str,
    output_path: Path,
) -> None:
    data: list[pd.Series] = []
    positions: list[int] = []
    labels: list[str] = []

    position = 1

    for feature in features:
        fruit = df[f"fruit_{feature}_mean"]
        background = df[f"background_{feature}_mean"]

        data.extend([fruit, background])
        positions.extend([position, position + 1])
        labels.extend(
            [
                f"{feature}\nFruto",
                f"{feature}\nFondo",
            ]
        )

        position += 3

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(
        data,
        positions=positions,
        tick_labels=labels,
    )

    ax.set_title(title)
    ax.set_ylabel("Valor normalizado")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_boxplot(
    df: pd.DataFrame,
    features: list[str],
    title: str,
    output_path: Path,
) -> None:
    data: list[pd.Series] = []
    positions: list[int] = []
    labels: list[str] = []

    position = 1

    for feature in features:
        ripe = df.loc[
            df["label"] == "ripe",
            f"fruit_{feature}_mean",
        ]

        unripe = df.loc[
            df["label"] == "unripe",
            f"fruit_{feature}_mean",
        ]

        data.extend([ripe, unripe])
        positions.extend([position, position + 1])
        labels.extend(
            [
                f"{feature}\nMaduro",
                f"{feature}\nInmaduro",
            ]
        )

        position += 3

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(
        data,
        positions=positions,
        tick_labels=labels,
    )

    ax.set_title(title)
    ax.set_ylabel("Valor normalizado")
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_class_histograms(
    df: pd.DataFrame,
    features: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(features),
        figsize=(15, 4),
    )

    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        column = f"fruit_{feature}_mean"

        ripe = df.loc[df["label"] == "ripe", column]
        unripe = df.loc[df["label"] == "unripe", column]

        ax.hist(
            ripe,
            bins=10,
            alpha=0.55,
            label="Maduro",
        )

        ax.hist(
            unripe,
            bins=10,
            alpha=0.55,
            label="Inmaduro",
        )

        ax.set_title(column)
        ax.set_xlabel("Valor normalizado")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.2)

    axes[0].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def create_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for feature in ["R", "G", "B", "H", "S", "V"]:
        column = f"fruit_{feature}_mean"

        ripe = df.loc[df["label"] == "ripe", column]
        unripe = df.loc[df["label"] == "unripe", column]

        ripe_mean = float(ripe.mean())
        unripe_mean = float(unripe.mean())

        rows.append(
            {
                "feature": feature,
                "ripe_mean": ripe_mean,
                "ripe_std": float(ripe.std()),
                "ripe_median": float(ripe.median()),
                "ripe_q1": float(ripe.quantile(0.25)),
                "ripe_q3": float(ripe.quantile(0.75)),
                "unripe_mean": unripe_mean,
                "unripe_std": float(unripe.std()),
                "unripe_median": float(unripe.median()),
                "unripe_q1": float(unripe.quantile(0.25)),
                "unripe_q3": float(unripe.quantile(0.75)),
                "absolute_mean_difference": abs(
                    ripe_mean - unripe_mean
                ),
            }
        )

    return pd.DataFrame(rows)


def create_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for feature in ["R", "G", "B", "H", "S", "V"]:
        fruit_column = f"fruit_{feature}_mean"
        background_column = f"background_{feature}_mean"

        fruit = df[fruit_column]
        background = df[background_column]

        fruit_mean = float(fruit.mean())
        background_mean = float(background.mean())

        rows.append(
            {
                "feature": feature,
                "fruit_mean": fruit_mean,
                "fruit_std": float(fruit.std()),
                "fruit_median": float(fruit.median()),
                "background_mean": background_mean,
                "background_std": float(background.std()),
                "background_median": float(background.median()),
                "absolute_mean_difference": abs(
                    fruit_mean - background_mean
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_input(df: pd.DataFrame) -> None:
    required_columns = {
        "image",
        "label",
    }

    for region in ("fruit", "background"):
        for feature in ("R", "G", "B", "H", "S", "V"):
            required_columns.add(
                f"{region}_{feature}_mean"
            )

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        raise ValueError("EDA input dataset is empty.")

    expected_labels = {"ripe", "unripe"}
    labels = set(df["label"].unique())

    if labels != expected_labels:
        raise ValueError(
            f"Unexpected labels: {sorted(labels)}. "
            f"Expected: {sorted(expected_labels)}"
        )

    if df[list(required_columns - {"image", "label"})].isna().any().any():
        raise ValueError("EDA input contains NaN values.")


def main() -> int:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)
    validate_input(df)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_region_boxplot(
        df=df,
        features=["R", "G", "B"],
        title="Comparación entre fruto y fondo - RGB",
        output_path=FIGURES_DIR / "fruit_vs_background_rgb.png",
    )

    save_region_boxplot(
        df=df,
        features=["H", "S", "V"],
        title="Comparación entre fruto y fondo - HSV",
        output_path=FIGURES_DIR / "fruit_vs_background_hsv.png",
    )

    save_class_boxplot(
        df=df,
        features=["R", "G", "B"],
        title="Tomates maduros e inmaduros - RGB",
        output_path=FIGURES_DIR / "ripe_vs_unripe_rgb.png",
    )

    save_class_boxplot(
        df=df,
        features=["H", "S", "V"],
        title="Tomates maduros e inmaduros - HSV",
        output_path=FIGURES_DIR / "ripe_vs_unripe_hsv.png",
    )

    save_class_histograms(
        df=df,
        features=["R", "G", "B"],
        title="Distribuciones de color - RGB",
        output_path=(
            FIGURES_DIR
            / "ripe_vs_unripe_histograms_rgb.png"
        ),
    )

    save_class_histograms(
        df=df,
        features=["H", "S", "V"],
        title="Distribuciones de color - HSV",
        output_path=(
            FIGURES_DIR
            / "ripe_vs_unripe_histograms_hsv.png"
        ),
    )

    class_summary = create_class_summary(df)
    class_summary.to_csv(
        TABLES_DIR / "ripe_vs_unripe_summary.csv",
        index=False,
    )

    region_summary = create_region_summary(df)
    region_summary.to_csv(
        TABLES_DIR / "fruit_vs_background_summary.csv",
        index=False,
    )

    print("EDA GENERATED")
    print(f"Samples: {len(df)}")
    print(
        f"Ripe:    {(df['label'] == 'ripe').sum()}"
    )
    print(
        f"Unripe:  {(df['label'] == 'unripe').sum()}"
    )

    print()
    print("FRUIT VS BACKGROUND")
    print(
        region_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("RIPE VS UNRIPE")
    print(
        class_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("OUTPUT")
    print(
        f"Figures: "
        f"{FIGURES_DIR.relative_to(PROJECT_ROOT)}"
    )
    print(
        f"Tables:  "
        f"{TABLES_DIR.relative_to(PROJECT_ROOT)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
