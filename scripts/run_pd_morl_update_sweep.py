"""Queue critic-update or sampling-cadence sweeps, at most two jobs per GPU.

Each run collects 2,000,960 base transitions using 160 environments x 4 steps.
Use --dry-run to preview. Repeat the same command to resume the same queue.
"""

import argparse
import math
from pathlib import Path
import shlex
import sys

import run_pd_morl_seed_scan as scan


def make_plans(seeds, critics, total):
    return {
        f"critic_{critic}_seed_{seed}": {
            "seed": seed,
            "overrides": {
                "total_timesteps": total,
                "rollout_length": 4,
                "critic_updates_per_rollout": critic,
                "+run_id": f"critic_{critic}_seed_{seed}",
            },
        }
        for seed in seeds
        for critic in critics
    }


def make_sampling_plans(seeds, lengths, total):
    # Keep the chosen baseline's critic/transition ratio at 320 / (160 * 4).
    return {
        f"rollout_{length}_seed_{seed}": {
            "seed": seed,
            "overrides": {
                "total_timesteps": total,
                "rollout_length": length,
                "critic_updates_per_rollout": 80 * length,
                "+run_id": f"rollout_{length}_seed_{seed}",
            },
        }
        for seed in seeds
        for length in lengths
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--critic-updates", nargs="+", type=int, default=[80, 160, 320])
    mode.add_argument("--sampling-lengths", nargs="+", type=int, choices=[2, 4, 8])
    parser.add_argument("--total-timesteps", type=int, default=2_000_960)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-jobs-per-gpu", type=int, choices=[1, 2], default=2)
    parser.add_argument("--cpus-per-job", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        not args.gpus
        or min(args.gpus) < 0
        or len(set(args.gpus)) != len(args.gpus)
        or min(args.seeds) < 0
        or max(args.seeds) >= 2**32
        or len(set(args.seeds)) != len(args.seeds)
        or min(args.critic_updates) < 1
        or len(set(args.critic_updates)) != len(args.critic_updates)
        or (
            args.sampling_lengths is not None
            and len(set(args.sampling_lengths)) != len(args.sampling_lengths)
        )
        or args.cpus_per_job < 1
        or not math.isfinite(args.min_free_gb)
        or args.min_free_gb < 0
        or any(
            not math.isfinite(x) or x <= 0
            for x in (args.poll_seconds, args.retry_delay_seconds)
        )
    ):
        parser.error(
            "require distinct valid GPUs/seeds/updates, positive CPUs/polling and nonnegative memory"
        )
    plans = (
        make_sampling_plans(args.seeds, args.sampling_lengths, args.total_timesteps)
        if args.sampling_lengths is not None
        else make_plans(args.seeds, args.critic_updates, args.total_timesteps)
    )
    for plan in plans.values():
        try:
            scan.expected_counters(plan["overrides"])
        except ValueError as error:
            parser.error(str(error))
    root = (scan.REPO / args.output_root).resolve()
    print(
        f"Jobs: {len(plans)}; GPUs: {args.gpus}; max jobs/GPU: {args.max_jobs_per_gpu}; "
        f"base transitions/job: {args.total_timesteps:,}",
        flush=True,
    )
    if args.dry_run:
        for name, plan in plans.items():
            print(
                shlex.join(
                    scan.training_command(
                        plan["seed"],
                        root / name / "attempt_001",
                        overrides=plan["overrides"],
                    )
                )
            )
        return 0
    if sys.platform != "linux":
        parser.error("run on Linux/lab4090; --dry-run works locally")
    return scan.run_scan(args, root, list(plans), plans)


if __name__ == "__main__":
    raise SystemExit(main())
