"""Fit on training, choose Youden on validation, then evaluate frozen test."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bayes import (
    POSITIVE_CLASS, evaluate_binary_classifier, log_likelihood_ratio,
    predict_with_threshold, select_youden_threshold,
)
from src.features import COLOR_FEATURES
from src.protocol import load_split, split_fingerprint, validate_partition_rows
from src.segmentation import load_selected_channels

INPUT_PATH = PROJECT_ROOT / "results" / "features" / "segmented_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "classification" / "bayes_analysis"
DECISION_PATH = OUTPUT_DIR / "decision.json"
# Reconfirmed from training EDA before the corrected final evaluation.
SELECTED_FEATURES = ["R", "G", "S"]
VAR_SMOOTHING = 1e-9


def validate_dataset(df: pd.DataFrame, split: pd.DataFrame, include_test: bool) -> None:
    partitions = ("training", "validation", "test") if include_test else (
        "training", "validation",
    )
    validate_partition_rows(df, split, partitions)
    if not set(COLOR_FEATURES).issubset(df.columns):
        raise ValueError("Missing candidate color features.")
    if not np.isfinite(df[list(COLOR_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite color features.")
    if set(df["segmentation_channels"]) != {load_selected_channels()}:
        raise ValueError("Features use a stale segmentation configuration.")


def save_evaluation(
    samples: pd.DataFrame, scores: np.ndarray, threshold: float, name: str,
    *, features: list[str] = SELECTED_FEATURES, output_dir: Path = OUTPUT_DIR,
) -> None:
    y = samples["label"].to_numpy()
    metrics = evaluate_binary_classifier(y, scores, threshold)
    record = {
        "dataset": name, "features": ",".join(features),
        "fit_partition": "training", "threshold_partition": "validation",
        "criterion": "Youden (largest finite threshold on ties)",
        **asdict(metrics),
    }
    pd.DataFrame([record]).to_csv(output_dir / f"{name}_metrics.csv", index=False)
    output = samples[["image", "label", "split", "segmentation_jaccard",
                      *features]].copy()
    output["score"] = scores
    output["prediction"] = predict_with_threshold(scores, threshold)
    output["correct"] = output["prediction"] == output["label"]
    output.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    fpr, tpr, thresholds = roc_curve(
        y, scores, pos_label=POSITIVE_CLASS, drop_intermediate=False,
    )
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        output_dir / f"{name}_roc.csv", index=False,
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"{name} (AUC = {metrics.auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Azar")
    ax.scatter([1 - metrics.specificity], [metrics.sensitivity],
               label="Umbral fijado en validation", zorder=5)
    role = "selección del umbral" if name == "validation" else "evaluación final"
    ax.set(title=f"Bayes {','.join(features)} — {name}: {role}",
           xlabel="Tasa de falsos positivos", ylabel="Sensibilidad",
           xlim=(0, 1), ylim=(0, 1.05))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_roc.png", dpi=200)
    plt.close(fig)
    print(pd.DataFrame([record]).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate test after verifying the saved frozen decision.")
    args = parser.parse_args()
    split = load_split()
    df = pd.read_csv(INPUT_PATH)
    # During development, the active feature table must contain no test rows.
    # Final test inference is enabled only by --evaluate-test.
    if args.evaluate_test:
        development = df.loc[df["split"].isin(["training", "validation"])].copy()
        validate_dataset(development, split, include_test=False)
    else:
        validate_dataset(df, split, include_test=False)
        development = df.copy()
    training = development.loc[development["split"] == "training"]
    validation = development.loc[development["split"] == "validation"]

    model = GaussianNB(var_smoothing=VAR_SMOOTHING)
    model.fit(training[SELECTED_FEATURES].to_numpy(), training["label"].to_numpy())
    validation_scores = log_likelihood_ratio(model, validation[SELECTED_FEATURES].to_numpy())
    threshold, youden, _, _ = select_youden_threshold(
        validation["label"].to_numpy(), validation_scores,
    )
    decision = {
        "features": SELECTED_FEATURES, "segmentation_channels": load_selected_channels(),
        "score": "log p(x | ripe) - log p(x | unripe)", "positive_class": POSITIVE_CLASS,
        "fit_partition": "training", "threshold_partition": "validation",
        "threshold": threshold, "youden": youden,
        "criterion": "maximum Youden; ties: largest finite threshold",
        "var_smoothing": VAR_SMOOTHING, "split_sha256": split_fingerprint(),
        "classes": model.classes_.tolist(), "theta": model.theta_.tolist(),
        "var": model.var_.tolist(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.evaluate_test:
        frozen = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
        if decision != frozen:
            raise ValueError("Pipeline differs from saved decision; test evaluation aborted.")
        validate_dataset(df, split, include_test=True)
        test = df.loc[df["split"] == "test"]
        scores = log_likelihood_ratio(model, test[SELECTED_FEATURES].to_numpy())
        save_evaluation(test, scores, threshold, "test")
    else:
        DECISION_PATH.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        save_evaluation(validation, validation_scores, threshold, "validation")
        print(f"Frozen decision: {DECISION_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
