"""Import completed PD-MORL training records into W&B without rerunning training."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from run_pd_morl_seed_scan import completed_result, read_json


def scalar_metrics(prefix, row):
    result = {}
    for key, value in row.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[f"{prefix}/{key}"] = value
        elif isinstance(value, list) and all(
            isinstance(x, (int, float)) for x in value
        ):
            result.update(
                {
                    f"{prefix}/{key}_{i}": x
                    for i, x in enumerate(value)
                    if math.isfinite(x)
                }
            )
    return result


def training_events(run_dir):
    """Merge recorded events at their actual base-transition counts."""
    events = {}

    def add(step, metrics):
        events.setdefault(int(step), {"env_steps": int(step)}).update(metrics)

    for line in (run_dir / "training_diagnostics.jsonl").read_text().splitlines():
        row = json.loads(line)
        add(row["base_inserts"], scalar_metrics("train", row))
    with (run_dir / "hv_history.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            metrics = {
                key: float(row[key])
                for key in (
                    "source_hv",
                    "mean_repeat_hv",
                    "source_sparsity",
                    "source_pareto_point_count",
                    "preference_count",
                    "repeats",
                    "wall_clock_seconds",
                )
            }
            add(row["actual_env_transitions"], scalar_metrics("eval", metrics))
    with (run_dir / "key_replacement_diagnostics.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            metrics = {
                key: float(row[key])
                for key in (
                    "candidate_r0",
                    "candidate_r1",
                    "stored_score",
                    "candidate_score",
                    "score_gap",
                    "stored_r0_after",
                    "stored_r1_after",
                )
            }
            metrics["replaced"] = int(row["replaced"].lower() == "true")
            add(
                row["actual_env_transitions"],
                scalar_metrics(f"key/{row['key_index']}", metrics),
            )
    final = read_json(run_dir / "final_training_eval/results.json")
    final_metrics = {
        key: final[key]
        for key in (
            "source_hv",
            "source_sparsity",
            "source_pareto_point_count",
            "mean_repeat_hv",
            "mean_repeat_sparsity",
            "hv_per_repeat",
            "sparsity_per_repeat",
        )
    }
    final_metrics.update(
        preference_count=len(final["preferences"]),
        repeats=len(final["returns_per_repeat"]),
    )
    add(final["actual_env_transitions"], scalar_metrics("final", final_metrics))
    return sorted(events.items()), final


def record_files(run_dir):
    files = [
        run_dir / name
        for name in (
            "run_metadata.json",
            "resolved_config.yaml",
            "counters_summary.json",
            "training_diagnostics.jsonl",
            "hv_history.csv",
            "hv_summary.json",
            "hv_curve.png",
            "key_replacement_diagnostics.csv",
            "pd_morl_evaluations.jsonl",
            "launcher.log",
            "final_training_eval/results.json",
            "final_training_eval/pareto_artifacts.npz",
        )
    ]
    files.extend(run_dir.glob("GPU-native*.log"))
    files.extend((run_dir / "pareto_fronts").glob("*.json"))
    return sorted(path for path in files if path.is_file())


def record_digest(run_dir, files):
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(run_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def completed_jobs(state, completed_only=False):
    jobs = state["jobs"]
    if not completed_only and (
        len(jobs) != len(state["contract"]["plans"])
        or any(job["status"] != "complete" for job in jobs.values())
    ):
        raise ValueError("all planned experiments must be complete before import")
    selected = {name: job for name, job in jobs.items() if job["status"] == "complete"}
    if not selected:
        raise ValueError("no completed experiments to import")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--project", default="pd-morl")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="publish finished jobs while the remaining jobs train",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    state = read_json(root / "scan.json")
    group = args.group or root.name
    prepared = []
    for name, job in completed_jobs(state, args.completed_only).items():
        source = Path(job["run_dir"])
        completed_result(source, job["seed"], job["overrides"])
        events, final = training_events(source)
        files = record_files(source)
        sha = record_digest(source, files)
        prepared.append((name, job, source, events, final, files, sha))
        print(
            json.dumps(
                dict(
                    name=name,
                    history_points=len(events),
                    files=len(files),
                    bytes=sum(p.stat().st_size for p in files),
                    source_hv=final["source_hv"],
                )
            ),
            flush=True,
        )
    if args.dry_run:
        return 0

    import wandb

    api = wandb.Api(timeout=30)
    entity = args.entity or api.default_entity
    existing = {
        run.id: run
        for run in api.runs(f"{entity}/{args.project}", filters={"group": group})
    }
    output = root.parent / "wandb_imports" / group
    output.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for name, job, source, events, final, files, sha in prepared:
        run_id = (
            "history-"
            + hashlib.sha256(
                f"{entity}/{args.project}/{group}/{name}".encode()
            ).hexdigest()[:16]
        )
        previous = existing.get(run_id)
        if previous is not None:
            if previous.config.get("record_sha256") != sha:
                raise ValueError(
                    f"source records changed for {name}; use another group"
                )
            if previous.state == "finished" and previous.summary.get("import_complete"):
                uploaded.append(dict(name=name, id=run_id, url=previous.url))
                continue
        metadata = read_json(source / "run_metadata.json")
        config = dict(metadata["frozen_hyperparameters"])
        config.update(
            record_sha256=sha,
            historical_import=True,
            source_directory=str(source),
            key_artifact_sha256=metadata["key_artifact_sha256"],
            key_objective_normalization=metadata["key_objective_normalization"],
            runtime_source_sha256=metadata["runtime_source_sha256"],
            original_git_commit=metadata["git_commit"],
            original_gpu_uuid=metadata["gpu_uuid"],
            original_physical_gpu_id=metadata["physical_gpu_id"],
        )
        with wandb.init(
            entity=entity,
            project=args.project,
            group=group,
            name=name,
            id=run_id,
            resume="allow",
            config=config,
            dir=str(output),
            job_type="historical-import",
            tags=[
                "pd-morl",
                f"{round(final['actual_env_transitions'] / 1_000_000)}m",
                "l2",
            ],
            settings=wandb.Settings(x_disable_stats=True, disable_git=True),
        ) as run:
            run.define_metric("env_steps")
            for prefix in ("train", "eval", "key", "final"):
                run.define_metric(f"{prefix}/*", step_metric="env_steps")
            for step, metrics in events:
                if step >= run.step:
                    run.log(metrics, step=step)
            plot_data = {
                "env_steps": final["actual_env_transitions"],
                "plots/hv_curve": wandb.Image(str(source / "hv_curve.png")),
                "plots/final_pareto": wandb.plot.scatter(
                    wandb.Table(
                        columns=["return_0", "return_1"], data=final["pareto_returns"]
                    ),
                    "return_0",
                    "return_1",
                    title="Final Pareto front: 1001 preferences x 3 repeats",
                ),
            }
            run.log(plot_data)
            artifact = wandb.Artifact(
                f"pd-morl-{name}-{sha[:8]}", type="training-records"
            )
            for path in files:
                artifact.add_file(str(path), name=path.relative_to(source).as_posix())
            run.log_artifact(artifact).wait()
            run.summary.update(
                {
                    **{
                        key: value
                        for key, value in events[-1][1].items()
                        if key.startswith("final/")
                    },
                    "original_started_utc": job["started_utc"],
                    "original_finished_utc": job["finished_utc"],
                    "import_history_points": len(events),
                    "import_source_files": len(files),
                    "import_complete": True,
                }
            )
            uploaded.append(dict(name=name, id=run_id, url=run.url))
        if record_digest(source, files) != sha:
            raise RuntimeError(f"original records changed during import: {name}")
        print("UPLOADED " + json.dumps(uploaded[-1]), flush=True)
    (output / "uploaded_runs.json").write_text(json.dumps(uploaded, indent=2) + "\n")
    print(
        json.dumps(
            dict(entity=entity, project=args.project, group=group, runs=uploaded)
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
