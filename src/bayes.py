from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted

POSITIVE_CLASS = "ripe"
NEGATIVE_CLASS = "unripe"
YOUDEN_TIE_ATOL = 1e-12


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    sensitivity: float
    specificity: float
    auc: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def log_likelihood_ratio(
    model: GaussianNB,
    X: np.ndarray,
    *,
    positive_class: str = POSITIVE_CLASS,
    negative_class: str = NEGATIVE_CLASS,
) -> np.ndarray:
    """log p(x | ripe) - log p(x | unripe), with no class-prior term.

    GaussianNB assumes conditionally independent features. Its fitted var_
    includes training-only variance smoothing, which is also used here.
    """
    check_is_fitted(model)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional array"
        )

    if X.shape[1] != model.theta_.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must have the fitted feature count and finite values.")
    if positive_class == negative_class:
        raise ValueError("Positive and negative classes must differ.")
    classes = list(model.classes_)

    if positive_class not in classes:
        raise ValueError(
            f"Positive class {positive_class!r} "
            "was not fitted by the model"
        )

    if negative_class not in classes:
        raise ValueError(
            f"Negative class {negative_class!r} "
            "was not fitted by the model"
        )

    positive_index = classes.index(
        positive_class
    )

    negative_index = classes.index(
        negative_class
    )

    def log_density(index: int) -> np.ndarray:
        variance = model.var_[index]
        if not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError("Fitted Gaussian variances must be positive and finite.")
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * variance)
            + (X - model.theta_[index]) ** 2 / variance,
            axis=1,
        )

    return log_density(positive_index) - log_density(negative_index)


def select_youden_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    positive_class: str = POSITIVE_CLASS,
) -> tuple[float, float, float, float]:
    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=positive_class,
        drop_intermediate=False,
    )

    youden_values = (
        tpr - fpr
    )

    # Exclude the all-negative sentinel. Ties: largest finite threshold,
    # favoring specificity; roc_curve returns thresholds in descending order.
    finite_indices = np.flatnonzero(np.isfinite(thresholds))
    best_youden = youden_values[finite_indices].max()
    # Equal rational TP/P - FP/N values can round to different floats.
    tied_indices = finite_indices[np.isclose(
        youden_values[finite_indices], best_youden, rtol=0, atol=YOUDEN_TIE_ATOL,
    )]
    best_index = int(tied_indices[0])

    threshold = float(
        thresholds[best_index]
    )

    sensitivity = float(
        tpr[best_index]
    )

    specificity = float(
        1.0 - fpr[best_index]
    )

    youden = float(
        youden_values[best_index]
    )

    return (
        threshold,
        youden,
        sensitivity,
        specificity,
    )


def predict_with_threshold(
    scores: np.ndarray,
    threshold: float,
    *,
    positive_class: str = POSITIVE_CLASS,
    negative_class: str = NEGATIVE_CLASS,
) -> np.ndarray:
    return np.where(
        scores >= threshold,
        positive_class,
        negative_class,
    )


def evaluate_binary_classifier(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    positive_class: str = POSITIVE_CLASS,
    negative_class: str = NEGATIVE_CLASS,
) -> BinaryMetrics:
    """
    Evaluate a binary classifier at a fixed decision threshold.

    The positive class is explicitly converted to 1 so that the ROC
    AUC is calculated using the same score direction as the Bayesian
    likelihood-ratio decision rule.
    """
    predictions = predict_with_threshold(
        scores=scores,
        threshold=threshold,
        positive_class=positive_class,
        negative_class=negative_class,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[
            negative_class,
            positive_class,
        ],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    y_binary = (
        y_true == positive_class
    ).astype(int)

    auc = roc_auc_score(
        y_binary,
        scores,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    return BinaryMetrics(
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        auc=float(auc),
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
