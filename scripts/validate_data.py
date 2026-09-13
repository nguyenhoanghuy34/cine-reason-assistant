#!/usr/bin/env python3
"""Run non-destructive validation for the MovieLens data layer."""

from pathlib import Path
import sys

# Allow both `python scripts/validate_data.py` and module invocation from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data_layer.loader import MovieLensDataLoader
from backend.data_layer.validation import validate_data


def main() -> int:
    report = validate_data(MovieLensDataLoader().load())
    print("Data validation report")
    for key, value in report.metrics.items():
        print(f"{key}: {value}")
    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"- {issue}")
        return 1
    print("No validation issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
