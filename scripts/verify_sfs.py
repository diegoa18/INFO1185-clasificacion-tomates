"""SFS leakage, determinism, tie rules and frozen-artifact regressions."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_bayes_analysis import INPUT_PATH, VAR_SMOOTHING, validate_dataset
from scripts.run_sfs import OUTPUT_DIR
from src.bayes import evaluate_binary_classifier, log_likelihood_ratio, select_youden_threshold
from src.features import COLOR_FEATURES
from src.protocol import load_split, split_fingerprint, validate_partition_rows
from src.sfs import select_sfs


def main() -> int:
    split = load_split()
    data = pd.read_csv(INPUT_PATH)
    validate_dataset(data, split, include_test=False)
    training = data.loc[data["split"] == "training"].sort_values(["label", "image"])
    validation = data.loc[data["split"] == "validation"].sort_values(["label", "image"])
    fits = []
    original_fit = GaussianNB.fit

    def record_fit(model, X, y, *args, **kwargs):
        fits.append((X.copy(), y.copy()))
        return original_fit(model, X, y, *args, **kwargs)

    with patch.object(GaussianNB, "fit", record_fit):
        result = select_sfs(data, split, var_smoothing=VAR_SMOOTHING)
    assert len(fits) == 21
    for (X, y), row in zip(fits, result.candidates.itertuples(index=False)):
        subset = row.features.split(",")
        np.testing.assert_array_equal(X, training[subset].to_numpy())
        np.testing.assert_array_equal(y, training["label"].to_numpy())
        model = GaussianNB(var_smoothing=VAR_SMOOTHING).fit(X, y)
        scores = log_likelihood_ratio(model, validation[subset].to_numpy())
        assert row.validation_auc == roc_auc_score(
            (validation["label"] == "ripe").astype(int), scores,
        )
    repeat = select_sfs(data.sample(frac=1, random_state=42), split)
    pd.testing.assert_frame_equal(result.candidates, repeat.candidates)
    pd.testing.assert_frame_equal(result.trajectory, repeat.trajectory)
    np.testing.assert_array_equal(result.model.theta_, repeat.model.theta_)
    assert result.features == repeat.features

    # Synthetic poison row: no real test values are read. Reject before any fit.
    poison = data.iloc[[0]].copy()
    poison["image"] = "synthetic-test"
    poison["split"] = "test"
    with patch.object(GaussianNB, "fit", side_effect=AssertionError("Unexpected fit")):
        try:
            select_sfs(pd.concat([data, poison], ignore_index=True), split)
        except ValueError:
            pass
        else:
            raise AssertionError("SFS accepted a test row")

    # Fixed synthetic equal AUCs test both tie rules independently of real scores.
    with patch("src.sfs.roc_auc_score", return_value=0.5):
        tied = select_sfs(data, split)
    assert tied.trajectory["added_feature"].tolist() == list(COLOR_FEATURES)
    assert tied.features == ["R"] and len(tied.trajectory) == 6
    # A tiny strict improvement must not be treated as an approximate AUC tie.
    with patch("src.sfs.roc_auc_score", side_effect=[0.5, 0.5 + 1e-13] + [0.5] * 19):
        near_tie = select_sfs(data, split)
    assert near_tie.features == ["G"]

    decision = json.loads((OUTPUT_DIR / "decision.json").read_text(encoding="utf-8"))
    assert decision["features"] == result.features
    assert decision["split_sha256"] == split_fingerprint()
    assert decision["development_features_sha256"] == sha256(INPUT_PATH.read_bytes()).hexdigest()
    assert decision["candidate_tie_order"] == list(COLOR_FEATURES)
    model = GaussianNB(var_smoothing=VAR_SMOOTHING).fit(
        training[result.features].to_numpy(), training["label"].to_numpy(),
    )
    np.testing.assert_array_equal(decision["theta"], model.theta_)
    np.testing.assert_array_equal(decision["var"], model.var_)
    scores = log_likelihood_ratio(model, validation[result.features].to_numpy())
    threshold, youden, _, _ = select_youden_threshold(validation["label"].to_numpy(), scores)
    assert decision["threshold"] == threshold and decision["youden"] == youden
    predictions = pd.read_csv(OUTPUT_DIR / "validation_predictions.csv")
    validate_partition_rows(predictions, split, ("validation",))
    np.testing.assert_allclose(predictions["score"], scores, rtol=1e-14, atol=1e-14)
    pd.testing.assert_frame_equal(pd.read_csv(OUTPUT_DIR / "candidate_metrics.csv"), result.candidates)
    pd.testing.assert_frame_equal(pd.read_csv(OUTPUT_DIR / "trajectory.csv"), result.trajectory)
    metrics = evaluate_binary_classifier(validation["label"].to_numpy(), scores, threshold)
    saved = pd.read_csv(OUTPUT_DIR / "validation_metrics.csv").iloc[0]
    for field in ("auc", "accuracy", "sensitivity", "specificity", "tn", "fp", "fn", "tp"):
        np.testing.assert_allclose(saved[field], getattr(metrics, field))
    print("PASS SFS: 21 training-only fits; validation-only AUC/Youden; no test rows;")
    print("exact ties, full trajectory, determinism and frozen model/artifacts verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
