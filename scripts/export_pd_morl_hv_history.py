"""Export training_full HV rows from existing PD-MORL JSONL snapshots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

FIELDS = (
    "seed", "run_id", "artifact_sha256", "kind", "preference_count", "repeats",
    "actual_env_transitions", "iteration", "source_hv", "mean_repeat_hv",
    "source_sparsity", "source_pareto_point_count", "wall_clock_seconds",
)


def export_history(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = input_path.parent / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    config_path = input_path.parent / "resolved_config.yaml"
    resolved = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else {}
    )
    seed = metadata.get("training_seed", resolved.get("seed", ""))
    artifact_sha = metadata.get("key_artifact_sha256", "")
    run_id = input_path.parent.name
    per_iteration = (
        int(resolved.get("rollout_length", 0))
        * int(resolved.get("num_envs", 0))
        * int(resolved.get("fold_iters", 1))
    )
    prefill = int(
        resolved.get(
            "learning_start_timesteps",
            resolved.get("source_prefill_env_transitions", 0),
        )
    )
    rows = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("kind") != "training_full":
            continue
        transitions = record.get(
            "actual_env_transitions", record.get("environment_transitions")
        )
        if transitions is None and per_iteration:
            transitions = prefill + int(record["iteration"]) * per_iteration
        if transitions is None:
            continue
        preferences = record.get("preferences", [])
        repeats = record.get("returns_per_repeat", [])
        source_hv = record.get("source_hv")
        if source_hv is None:
            source_hv = record.get("final_hv", record.get("mean_hv"))
        source_sparsity = record.get("source_sparsity")
        if source_sparsity is None:
            source_sparsity = record.get(
                "final_sparsity", record.get("mean_sparsity")
            )
        row = {
            "seed": seed,
            "run_id": run_id,
            "artifact_sha256": artifact_sha,
            "kind": "training_full",
            "preference_count": len(preferences),
            "repeats": len(repeats),
            "actual_env_transitions": int(transitions),
            "iteration": int(record["iteration"]),
            "source_hv": source_hv,
            "mean_repeat_hv": record.get("mean_repeat_hv", record.get("mean_hv")),
            "source_sparsity": source_sparsity,
            "source_pareto_point_count": record.get(
                "source_pareto_point_count", record.get("final_pareto_point_count", len(record.get("pareto_returns", [])))
            ),
            "wall_clock_seconds": record.get("wall_clock_seconds"),
        }
        rows[(run_id, "training_full", row["actual_env_transitions"])] = row
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: item["actual_env_transitions"]):
            writer.writerow(row)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"exported_rows={export_history(args.input, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
