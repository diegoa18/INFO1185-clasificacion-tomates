from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".fig", ".tiff"}

@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    label: str


def _find_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def discover_samples(dataset_dir: Path) -> list[Sample]:
    samples: list[Sample] = []

    for label in ("ripe", "unripe"):
        class_dir = dataset_dir / label
        image_dir = class_dir / "images"
        mask_dir = class_dir / "masks"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"no esta el coso este: {image_dir}")

        if not mask_dir.is_dir():
            raise FileNotFoundError(f"no esta el coso este: {mask_dir}")

        for image_path in _find_images(image_dir):
            mask_path = mask_dir / f"{image_path.stem}_mask.png"

            samples.append(Sample(image_path=image_path, mask_path=mask_path, label=label))

    return samples


def validate_sample(sample: Sample) -> list[str]:
    errors: list[str] = []

    if not sample.image_path.is_file():
        errors.append(f"no esta la imagen: {sample.image_path}")

    if not sample.mask_path.is_file():
        errors.append(f"no esta la mascara: {sample.mask_path}")

    if errors:
        return errors

    try:
        with Image.open(sample.image_path) as image:
            image.verify()

        with Image.open(sample.mask_path) as mask:
            mask.verify()

    except Exception as exc:
        errors.append(f"imagen o mascara invalida para {sample.image_path.name}: {exc}")
        return errors

    try:
        with Image.open(sample.image_path) as image, Image.open(sample.mask_path) as mask:
            if image.size != mask.size:
                errors.append(
                    f"tamaño incompatible para {sample.image_path.name}: "
                    f"image={image.size}, mask={mask.size}"
                )
    except Exception as exc:
        errors.append(f"no se comparan tamaños para {sample.image_path.name}: {exc}")

    return errors
