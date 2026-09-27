"""Aggregate final evaluation JSON files without mixing training state."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _find_metrics(run_dir: Path):
    for name in ("final_eval.json", "metrics.json", "results.json"):
        path = run_dir / "final_eval" / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _value(data, *names):
    lowered = {str(key).lower(): value for key, value in data.items()}
    for name in names:
        if name in lowered:
            return float(lowered[name])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for run_dir in sorted(args.root.glob("walker/seed_*")):
        try:
            seed = int(run_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        metrics = _find_metrics(run_dir)
        if not metrics:
            continue
        rows.append(
            {
                "training_seed": seed,
                "HV": _value(metrics, "hv", "hypervolume", "eval/hypervolume"),
                "sparsity": _value(metrics, "sparsity", "eval/sparsity"),
                "pareto_point_count": _value(metrics, "pareto_point_count"),
            }
        )
    summary = {"runs": rows, "count": len(rows)}
    for key in ("HV", "sparsity"):
        values = [row[key] for row in rows if row[key] is not None]
        if values:
            summary[f"{key}_mean"] = statistics.mean(values)
            summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    output = args.output or args.root / "aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
