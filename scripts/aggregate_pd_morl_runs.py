"""Aggregate same-protocol PD-MORL HV histories across seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "seed", "run_id", "artifact_sha256", "kind", "preference_count", "repeats",
    "actual_env_transitions", "iteration", "source_hv", "mean_repeat_hv",
    "source_sparsity", "source_pareto_point_count", "wall_clock_seconds",
)


def _history_rows(root: Path) -> list[dict]:
    rows = {}
    for path in root.rglob("hv_history.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row.get("run_id", path.parent.name),
                    row.get("kind", ""),
                    int(row["actual_env_transitions"]),
                )
                row["run_id"] = key[0]
                rows[key] = row
    return sorted(
        rows.values(),
        key=lambda row: (
            int(row.get("seed", -1) or -1),
            int(row["actual_env_transitions"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = _history_rows(args.root)
    if not rows:
        raise SystemExit("no hv_history.csv files found")
    output = args.output or args.root / "hv_history_all_seeds.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
    summary = {
        "rows": len(rows),
        "runs": sorted({row["run_id"] for row in rows}),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
