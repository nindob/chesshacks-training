from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/log_training_metrics.py <metrics_json> <output_csv>")
        sys.exit(1)
    metrics_json = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])
    with metrics_json.open() as f:
        metrics = json.load(f)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "avg_policy_loss", "avg_value_loss", "avg_total_loss"])
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


if __name__ == "__main__":
    main()
