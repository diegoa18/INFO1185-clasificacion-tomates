from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import discover_samples, validate_sample

DATASET_DIR = PROJECT_ROOT / "dataset"

def main() -> int:
    samples = discover_samples(DATASET_DIR)

    if not samples:
        print("ERROR: no samples found.")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    label_counts = Counter()

    for sample in samples:
        label_counts[sample.label] += 1
        errors.extend(validate_sample(sample))

    print("DATASET AUDIT")
    print(f"Total samples: {len(samples)}")
    print(f"Ripe:         {label_counts['ripe']}")
    print(f"Unripe:       {label_counts['unripe']}")
    print()

    if errors:
        print("ERRORS")
        for error in errors:
            print(f"- {error}")
        print()
    else:
        print("Image/mask validation: OK")
        print()

    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(f"- {warning}")
        print()

    if errors:
        print("Dataset audit FAILED.")
        return 1

    print("Dataset audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
