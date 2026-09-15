"""Freeze SFS + GaussianNB using only the active development feature table."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bayes_analysis import (
    INPUT_PATH, VAR_SMOOTHING, save_evaluation, validate_dataset,
)
from src.bayes import POSITIVE_CLASS, YOUDEN_TIE_ATOL, select_youden_threshold
from src.features import COLOR_FEATURES
from src.protocol import load_split, split_fingerprint
from src.segmentation import load_selected_channels
from src.sfs import select_sfs

OUTPUT_DIR = PROJECT_ROOT / "results/classification/bayes_sfs"


def main() -> int:
    split = load_split()
    data = pd.read_csv(INPUT_PATH)
    validate_dataset(data, split, include_test=False)
    result = select_sfs(data, split, var_smoothing=VAR_SMOOTHING)
    validation = data.loc[data["split"] == "validation"].sort_values(["label", "image"])
    threshold, youden, _, _ = select_youden_threshold(
        validation["label"].to_numpy(), result.validation_scores,
    )
    decision = {
        "pipeline": "SFS + GaussianNB", "features": result.features,
        "candidate_features": list(COLOR_FEATURES),
        "selection_criterion": "maximum validation AUC at each forward addition",
        "candidate_tie_order": list(COLOR_FEATURES), "auc_ties": "exact equality",
        "trajectory": "all sizes 1 through 6; no early stopping",
        "final_subset_rule": "maximum trajectory AUC; ties: fewest features",
        "fit_partition": "training", "selection_partition": "validation",
        "threshold_partition": "validation", "positive_class": POSITIVE_CLASS,
        "score": "log p(x | ripe) - log p(x | unripe)",
        "threshold": threshold, "youden": youden,
        "threshold_criterion": "maximum Youden; ties: largest finite threshold",
        "youden_tie_atol": YOUDEN_TIE_ATOL, "var_smoothing": VAR_SMOOTHING,
        "priors": None, "split_sha256": split_fingerprint(),
        "development_features_sha256": sha256(INPUT_PATH.read_bytes()).hexdigest(),
        "segmentation_channels": load_selected_channels(),
        "classes": result.model.classes_.tolist(),
        "theta": result.model.theta_.tolist(), "var": result.model.var_.tolist(),
        "class_count": result.model.class_count_.tolist(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.candidates.to_csv(OUTPUT_DIR / "candidate_metrics.csv", index=False)
    result.trajectory.to_csv(OUTPUT_DIR / "trajectory.csv", index=False)
    (OUTPUT_DIR / "decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    save_evaluation(validation, result.validation_scores, threshold, "validation",
                    features=result.features, output_dir=OUTPUT_DIR)
    print(result.trajectory.to_string(index=False))
    print(f"Frozen SFS features: {','.join(result.features)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
