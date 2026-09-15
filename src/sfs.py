"""Full forward trajectory, training-only fits and validation-AUC selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB

from src.bayes import POSITIVE_CLASS, log_likelihood_ratio
from src.features import COLOR_FEATURES
from src.protocol import validate_partition_rows


@dataclass
class SFSResult:
    features: list[str]
    model: GaussianNB
    validation_scores: np.ndarray
    candidates: pd.DataFrame
    trajectory: pd.DataFrame


def select_sfs(
    data: pd.DataFrame, split: pd.DataFrame, *, var_smoothing: float = 1e-9,
) -> SFSResult:
    """Exact AUC ties: candidate order; final ties: smallest subset.

    The whole input must match training+validation. Test rows are rejected,
    rather than silently discarded. Only the six color columns enter models.
    """
    validate_partition_rows(data, split, ("training", "validation"))
    if not np.isfinite(data[list(COLOR_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("SFS requires finite color features.")
    data = data.sort_values(["label", "image"])
    training = data.loc[data["split"] == "training"]
    validation = data.loc[data["split"] == "validation"]
    y_training = training["label"].to_numpy()
    y_validation = (validation["label"].to_numpy() == POSITIVE_CLASS).astype(int)
    selected: list[str] = []
    candidates: list[dict] = []
    trajectory: list[dict] = []
    best_auc = -np.inf
    final_features: list[str] = []
    final_model = None
    final_scores = None

    for stage in range(1, len(COLOR_FEATURES) + 1):
        stage_auc = -np.inf
        stage_records = []
        for feature in COLOR_FEATURES:
            if feature in selected:
                continue
            subset = [*selected, feature]
            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(training[subset].to_numpy(), y_training)
            scores = log_likelihood_ratio(model, validation[subset].to_numpy())
            auc = float(roc_auc_score(y_validation, scores))
            record = {"stage": stage, "added_feature": feature,
                      "features": ",".join(subset), "validation_auc": auc,
                      "fit_partition": "training", "selection_partition": "validation"}
            stage_records.append(record)
            # Strict comparison preserves the first candidate on exact ties.
            if auc > stage_auc:
                stage_auc = auc
                winner = feature
                stage_model = model
                stage_scores = scores
        selected = [*selected, winner]
        for record in stage_records:
            record["chosen_addition"] = record["added_feature"] == winner
        candidates.extend(stage_records)
        trajectory.append({"stage": stage, "added_feature": winner,
                           "features": ",".join(selected), "validation_auc": stage_auc})
        # Stages are increasing in size; strict > keeps the smallest maximum.
        if stage_auc > best_auc:
            best_auc = stage_auc
            final_features = selected.copy()
            final_model = stage_model
            final_scores = stage_scores.copy()

    assert final_model is not None and final_scores is not None
    return SFSResult(final_features, final_model, final_scores,
                     pd.DataFrame(candidates), pd.DataFrame(trajectory))
