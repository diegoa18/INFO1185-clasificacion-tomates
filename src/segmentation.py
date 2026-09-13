from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans


CHANNEL_INDICES = {
    "R": (0,),
    "G": (1,),
    "B": (2,),
    "RG": (0, 1),
    "RB": (0, 2),
    "GB": (1, 2),
    "RGB": (0, 1, 2),
}

CHANNEL_COMBINATIONS = tuple(CHANNEL_INDICES)


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    tomato_cluster: int
    border_occupancy_cluster_0: float
    border_occupancy_cluster_1: float
    inertia: float
    n_iter: int


def _build_border_mask(
    height: int,
    width: int,
    fraction: float = 0.05,
) -> np.ndarray:
    if not 0 < fraction < 0.5:
        raise ValueError(
            "border fraction must be between 0 and 0.5"
        )

    border_width = max(
        1,
        int(round(min(height, width) * fraction)),
    )

    border = np.zeros(
        (height, width),
        dtype=bool,
    )

    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    return border


def _cluster_border_occupancy(
    labels: np.ndarray,
    cluster: int,
    border_mask: np.ndarray,
) -> float:
    border_size = int(border_mask.sum())

    if border_size == 0:
        raise ValueError(
            "Border mask is empty."
        )

    cluster_on_border = int(
        np.logical_and(
            labels == cluster,
            border_mask,
        ).sum()
    )

    return cluster_on_border / border_size


def select_tomato_cluster(
    labels: np.ndarray,
    border_fraction: float = 0.05,
) -> tuple[int, float, float]:
    if labels.ndim != 2:
        raise ValueError(
            "labels must be a two-dimensional array"
        )

    height, width = labels.shape

    border_mask = _build_border_mask(
        height=height,
        width=width,
        fraction=border_fraction,
    )

    occupancy_0 = _cluster_border_occupancy(
        labels=labels,
        cluster=0,
        border_mask=border_mask,
    )

    occupancy_1 = _cluster_border_occupancy(
        labels=labels,
        cluster=1,
        border_mask=border_mask,
    )

    if np.isclose(
        occupancy_0,
        occupancy_1,
    ):
        tomato_cluster = int(
            labels[
                height // 2,
                width // 2,
            ]
        )
    elif occupancy_0 < occupancy_1:
        tomato_cluster = 0
    else:
        tomato_cluster = 1

    return (
        tomato_cluster,
        occupancy_0,
        occupancy_1,
    )


def segment_kmeans(
    rgb: np.ndarray,
    channels: str,
    *,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    border_fraction: float = 0.05,
) -> SegmentationResult:
    if channels not in CHANNEL_INDICES:
        raise ValueError(
            f"Invalid channel combination: {channels}"
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "rgb must have shape (height, width, 3)"
        )

    if not np.isfinite(rgb).all():
        raise ValueError(
            "rgb contains non-finite values"
        )

    indices = CHANNEL_INDICES[channels]

    height, width, _ = rgb.shape

    pixels = rgb[:, :, indices].reshape(
        height * width,
        len(indices),
    )

    model = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )

    flat_labels = model.fit_predict(
        pixels
    )

    labels = flat_labels.reshape(
        height,
        width,
    )

    (
        tomato_cluster,
        occupancy_0,
        occupancy_1,
    ) = select_tomato_cluster(
        labels=labels,
        border_fraction=border_fraction,
    )

    tomato_mask = (
        labels == tomato_cluster
    )

    return SegmentationResult(
        mask=tomato_mask,
        tomato_cluster=tomato_cluster,
        border_occupancy_cluster_0=occupancy_0,
        border_occupancy_cluster_1=occupancy_1,
        inertia=float(model.inertia_),
        n_iter=int(model.n_iter_),
    )


def jaccard_index(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> float:
    if predicted.shape != reference.shape:
        raise ValueError(
            "Predicted and reference masks must have "
            "the same shape"
        )

    predicted_bool = predicted.astype(bool)
    reference_bool = reference.astype(bool)

    intersection = int(
        np.logical_and(
            predicted_bool,
            reference_bool,
        ).sum()
    )

    union = int(
        np.logical_or(
            predicted_bool,
            reference_bool,
        ).sum()
    )

    if union == 0:
        return 1.0

    return intersection / union
