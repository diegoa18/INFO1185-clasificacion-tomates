from __future__ import annotations
from pathlib import Path
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image


COLOR_FEATURES = ("R", "G", "B", "H", "S", "V")
STATISTICS = ("mean", "std", "median", "q1", "q3")


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        ) / 255.0

    return rgb


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as mask:
        mask_array = np.asarray(
            mask.convert("L")
        )

    foreground = mask_array > 0

    if not np.any(foreground):
        raise ValueError(
            f"Empty foreground mask: {path}"
        )

    if np.all(foreground):
        raise ValueError(
            f"Mask contains no background pixels: {path}"
        )

    return foreground


def convert_rgb_to_hsv(
    rgb: np.ndarray,
) -> np.ndarray:
    return rgb_to_hsv(rgb)


def _validate_rgb_and_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "rgb must have shape (height, width, 3)"
        )

    if mask.ndim != 2:
        raise ValueError(
            "mask must be two-dimensional"
        )

    if rgb.shape[:2] != mask.shape:
        raise ValueError(
            "rgb and mask dimensions do not match"
        )

    if not np.isfinite(rgb).all():
        raise ValueError(
            "rgb contains non-finite values"
        )

    if not np.any(mask):
        raise ValueError(
            "mask contains no foreground pixels"
        )


def extract_mean_color_features(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    mask = mask.astype(bool)

    _validate_rgb_and_mask(
        rgb=rgb,
        mask=mask,
    )

    hsv = convert_rgb_to_hsv(
        rgb
    )

    channels = {
        "R": rgb[:, :, 0],
        "G": rgb[:, :, 1],
        "B": rgb[:, :, 2],
        "H": hsv[:, :, 0],
        "S": hsv[:, :, 1],
        "V": hsv[:, :, 2],
    }

    return {
        feature: float(
            channel[mask].mean()
        )
        for feature, channel
        in channels.items()
    }


def _summary(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q1": float(
            np.percentile(values, 25)
        ),
        "q3": float(
            np.percentile(values, 75)
        ),
    }


def extract_color_statistics(
    image_path: Path,
    mask_path: Path,
) -> dict[str, float]:
    rgb = load_rgb_image(
        image_path
    )

    mask = load_mask(
        mask_path
    )

    if rgb.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask shape mismatch: "
            f"{image_path} -> {rgb.shape[:2]}, "
            f"{mask_path} -> {mask.shape}"
        )

    hsv = convert_rgb_to_hsv(
        rgb
    )

    data = {
        "R": rgb[:, :, 0],
        "G": rgb[:, :, 1],
        "B": rgb[:, :, 2],
        "H": hsv[:, :, 0],
        "S": hsv[:, :, 1],
        "V": hsv[:, :, 2],
    }

    regions = {
        "fruit": mask,
        "background": ~mask,
    }

    result: dict[str, float] = {}

    for region_name, region_mask in regions.items():
        for feature_name, channel in data.items():
            values = channel[
                region_mask
            ]

            statistics = _summary(
                values
            )

            for statistic_name, value in statistics.items():
                result[
                    f"{region_name}_"
                    f"{feature_name}_"
                    f"{statistic_name}"
                ] = value

    return result
