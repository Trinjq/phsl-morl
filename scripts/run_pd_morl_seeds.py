"""Run independent single-GPU PD-MORL experiments with a small GPU pool."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _run(repo: Path, gpu: int, seed: int, run_name: str, args: argparse.Namespace) -> int:
    run_dir = repo / args.output_root / "walker" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    command = [
        sys.executable,
        str(repo / "scripts" / "train.py"),
        "env=brax/walker2d",
        "agent=mo-td3",
        f"seed={seed}",
        f"hydra.run.dir={run_dir}",
        f"output_dir={run_dir}",
    ] + list(args.overrides)
    print(
        f"run_id=pd_morl_walker_{run_name} physical_gpu={gpu} "
        f"training_seed={seed} CUDA_VISIBLE_DEVICES={gpu} output_dir={run_dir}",
        flush=True,
    )
    completed = subprocess.run(command, cwd=repo, env=env, check=False)
    return completed.returncode


def _verify_reports(repo: Path, args: argparse.Namespace) -> int:
    reports = []
    for index, seed in enumerate(args.seeds):
        run_name = (
            f"seed_{seed}_rep_{index}"
            if Counter(args.seeds)[seed] > 1
            else f"seed_{seed}"
        )
        path = repo / args.output_root / "walker" / run_name / "smoke_report.json"
        if not path.exists():
            print(f"missing smoke report: {path}", file=sys.stderr)
            return 1
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    required = (
        "replay_sampling", "critic_update_count", "actor_update_count",
        "target_update_count", "her_active", "loss_finite", "checkpoint_path",
    )
    for report in reports:
        if any(not report.get(field) for field in required):
            print(f"incomplete learner smoke report: {report}", file=sys.stderr)
            return 1
        if report["jax_visible_device_count"] != 1:
            print(f"multi-GPU child detected: {report}", file=sys.stderr)
            return 1
    if len({report["physical_gpu_uuid"] for report in reports}) != len(reports):
        print("physical GPU UUID collision", file=sys.stderr)
        return 1
    if len({report["checkpoint_path"] for report in reports}) != len(reports):
        print("checkpoint path collision", file=sys.stderr)
        return 1
    if any(not report.get("interpolator_fingerprint") for report in reports):
        print("missing interpolator fingerprint", file=sys.stderr)
        return 1
    for field in ("model_fingerprint", "optimizer_fingerprint", "replay_fingerprint"):
        if len({report[field] for report in reports}) != len(reports):
            print(f"{field} collision", file=sys.stderr)
            return 1
    summary = {"status": "PASS", "runs": reports}
    summary_path = repo / args.output_root / "independent_seed_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"independent_seed_smoke_summary={summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", default="outputs/pd_morl")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-learner-smoke", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if not args.gpus:
        raise SystemExit("at least one GPU is required")
    workers = min(len(args.gpus), args.max_parallel or len(args.gpus))
    duplicate_seeds = Counter(args.seeds)
    if args.dry_run:
        for index, seed in enumerate(args.seeds):
            suffix = f"_rep_{index}" if duplicate_seeds[seed] > 1 else ""
            print(f"gpu={args.gpus[index % len(args.gpus)]} seed={seed}{suffix}")
        return 0
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run,
                repo,
                args.gpus[index % len(args.gpus)],
                seed,
                f"seed_{seed}_rep_{index}" if duplicate_seeds[seed] > 1 else f"seed_{seed}",
                args,
            ): seed
            for index, seed in enumerate(args.seeds)
        }
        for future in as_completed(futures):
            seed = futures[future]
            code = future.result()
            if code:
                failures.append((seed, code))
    if failures:
        print(f"failed_runs={failures}", file=sys.stderr)
        return 1
    if args.require_learner_smoke:
        return _verify_reports(repo, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
