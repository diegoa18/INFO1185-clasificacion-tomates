"""Regression checks for the split, likelihood ratio and training-only fit."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bayes import (
    evaluate_binary_classifier, log_likelihood_ratio, select_youden_threshold,
)
from src.protocol import correct_split, load_split, validate_partition_rows
from src.segmentation import load_selected_channels
from scripts.run_bayes_analysis import SELECTED_FEATURES, VAR_SMOOTHING
from scripts.create_split import archive_legacy_results


def check_score() -> None:
    # Unequal priors expose the former posterior-odds bug.
    X = np.array([[0.0], [0.2], [0.3], [0.4], [2.0], [2.2]])
    y = np.array(["unripe"] * 4 + ["ripe"] * 2)
    model = GaussianNB().fit(X, y)
    probe = np.array([[0.1], [1.0], [2.1]])
    score = log_likelihood_ratio(model, probe)
    positive = list(model.classes_).index("ripe")
    negative = list(model.classes_).index("unripe")
    posterior = model.predict_log_proba(probe)
    reference = posterior[:, positive] - posterior[:, negative] - np.log(
        model.class_prior_[positive] / model.class_prior_[negative]
    )
    np.testing.assert_allclose(score, reference)
    other_priors = GaussianNB(priors=[0.9, 0.1]).fit(X, y)
    np.testing.assert_allclose(score, log_likelihood_ratio(other_priors, probe))
    np.testing.assert_allclose(-score, log_likelihood_ratio(
        model, probe, positive_class="unripe", negative_class="ripe",
    ))
    scores = log_likelihood_ratio(model, X)
    threshold, youden, _, _ = select_youden_threshold(y, scores)
    metrics = evaluate_binary_classifier(y, scores, threshold)
    assert metrics.auc == 1.0 and metrics.tp == 2 and metrics.tn == 4
    assert metrics.accuracy == 1.0 and youden == 1.0
    assert np.isfinite(threshold)
    inverted = evaluate_binary_classifier(y, -scores, 0.0)
    assert inverted.auc == 0.0
    # 4/6 - 0/6 and 5/6 - 1/6 are an exact mathematical tie despite rounding.
    tied_y = np.array(["ripe"] * 4 + ["unripe", "ripe"] + ["unripe"] * 5 + ["ripe"])
    tied_scores = np.arange(12, 0, -1, dtype=float)
    assert select_youden_threshold(tied_y, tied_scores)[0] == 9.0


def check_legacy_archive() -> None:
    # Simulate receiving the corrected split via Git while ignored results remain.
    with TemporaryDirectory(prefix="tomato-protocol-") as directory:
        results = Path(directory) / "results"
        features = results / "features/segmented_features.csv"
        features.parent.mkdir(parents=True)
        historical = "image,label,split\nfixture,ripe,development\n"
        features.write_text(historical, encoding="utf-8")
        assert archive_legacy_results(results)  # No legacy split is required.
        archived = results / "legacy_development_test/features/segmented_features.csv"
        assert archived.read_text(encoding="utf-8") == historical
        assert not features.exists()
        features.parent.mkdir()
        features.write_text("image,label,split\nfixture,ripe,training\n", encoding="utf-8")
        assert not archive_legacy_results(results)
        assert features.exists() and archived.read_text(encoding="utf-8") == historical


def main() -> int:
    check_score()
    check_legacy_archive()
    split = load_split()
    pd.testing.assert_frame_equal(split, correct_split(split.sample(frac=1, random_state=7)))
    legacy = split.copy()
    legacy.loc[legacy["split"] != "test", "split"] = "development"
    pd.testing.assert_frame_equal(split, correct_split(legacy))
    tampered = split.copy()
    tampered.loc[tampered["split"] == "test", "split"] = "training"
    try:
        correct_split(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("Moving former test images was accepted.")

    eda = pd.read_csv(PROJECT_ROOT / "results/eda/tomato_features.csv")
    validate_partition_rows(eda, split, ("training",))
    summary = pd.read_csv(PROJECT_ROOT / "results/segmentation/jaccard_summary.csv")
    validation_summary = summary.loc[summary["split"] == "validation"]
    best = validation_summary.loc[validation_summary["mean_jaccard"].idxmax(), "channels"]
    assert load_selected_channels() == best
    features = pd.read_csv(PROJECT_ROOT / "results/features/segmented_features.csv")
    validate_partition_rows(features, split, ("training", "validation"))
    training = features.loc[features["split"] == "training"]
    validation = features.loc[features["split"] == "validation"]
    model = GaussianNB(var_smoothing=VAR_SMOOTHING).fit(
        training[SELECTED_FEATURES].to_numpy(), training["label"].to_numpy(),
    )
    decision = json.loads((PROJECT_ROOT / "results/classification/bayes_analysis/decision.json")
                          .read_text(encoding="utf-8"))
    np.testing.assert_allclose(decision["theta"], model.theta_, rtol=0, atol=0)
    np.testing.assert_allclose(decision["var"], model.var_, rtol=0, atol=0)
    scores = log_likelihood_ratio(model, validation[SELECTED_FEATURES].to_numpy())
    threshold, _, _, _ = select_youden_threshold(validation["label"].to_numpy(), scores)
    assert decision["threshold"] == threshold
    predictions = pd.read_csv(
        PROJECT_ROOT / "results/classification/bayes_analysis/validation_predictions.csv"
    )
    validate_partition_rows(predictions, split, ("validation",))
    np.testing.assert_allclose(predictions["score"], scores)
    print("PASS: frozen test membership, deterministic 37/12/13, training-only EDA/fit,")
    print("validation-only segmentation/threshold, prior-free score and ripe-positive ROC/AUC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
