"""Plot the existing 201x3 PD-MORL HV history without evaluating."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_hv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("kind") == "training_full"
            and row.get("preference_count") == "201"
            and row.get("repeats") == "3"
        ]
    rows.sort(key=lambda row: int(row["actual_env_transitions"]))
    return rows


def plot_hv_csv(input_path: Path, output_path: Path) -> dict:
    rows = load_hv_rows(input_path)
    if not rows:
        raise ValueError(f"no training_full 201x3 rows in {input_path}")
    transitions = [int(row["actual_env_transitions"]) for row in rows]
    source_hv = [float(row["source_hv"]) for row in rows]
    if not all(math.isfinite(value) for value in source_hv):
        raise ValueError("HV history contains non-finite source_hv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(transitions, source_hv, marker="o", linewidth=1.5, label="source_hv")
    axis.set_xlabel("Environment Transitions")
    axis.set_ylabel("Hypervolume")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    summary = {
        "kind": "training_full",
        "preference_count": 201,
        "repeats": 3,
        "points": len(rows),
        "first_source_hv": source_hv[0],
        "last_source_hv": source_hv[-1],
        "peak_source_hv": max(source_hv),
        "first_actual_env_transitions": transitions[0],
        "last_actual_env_transitions": transitions[-1],
    }
    output_path.with_name("hv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot_hv_csv(args.input, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
