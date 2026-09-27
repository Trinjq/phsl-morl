"""Run independent single-GPU PD-MORL experiments with a small GPU pool."""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", default="outputs/pd_morl")
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
