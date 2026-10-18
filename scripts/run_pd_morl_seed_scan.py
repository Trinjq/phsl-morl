"""Queue independent, full-budget PD-MORL seeds on lab4090 (Linux).

Default seeds: 0, 7, ..., 553. Preview with --dry-run before starting.
Re-run the same command to skip completed seeds and resume interrupted ones.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TOTAL = 10_003_520
ITERATIONS = 7_812
EXPECTED = {
    "environment_step_count": TOTAL,
    "inner_rollout_count": ITERATIONS,
    "host_chunk_count": 1953,
    "critic_optimizer_step_count": 4_999_680,
    "actor_optimizer_step_count": 499_968,
    "target_update_count": 499_968,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_digest():
    digest = hashlib.sha256()
    for folder in ("evorl", "scripts", "configs"):
        for path in sorted((REPO / folder).rglob("*")):
            # Scheduler maintenance must not invalidate unchanged learner state.
            if path in {REPO / "scripts/run_pd_morl_seed_scan.py",
                        REPO / "scripts/run_pd_morl_update_sweep.py"}:
                continue
            if path.is_file() and path.suffix in {".py", ".yaml", ".txt", ".json"}:
                digest.update(path.relative_to(REPO).as_posix().encode())
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def latest_checkpoint(seed_dir):
    candidates = [
        path for path in seed_dir.glob("attempt_*/checkpoints/*")
        if path.name.isdigit() and (path / "default/_METADATA").is_file()
    ]
    return max(candidates, key=lambda p: (int(p.name), p.parent.parent.name), default=None)


def training_command(seed, run_dir, checkpoint=None, overrides=None):
    settings = {
        "seed": seed, "total_timesteps": TOTAL, "formal_reproduction": "false",
        "key_diagnostics.enable": "true", "+run_id": f"pd_morl_seed_{seed}",
        "hydra.run.dir": run_dir,
    }
    settings.update(overrides or {})
    command = [
        sys.executable, str(REPO / "scripts/train.py"),
        "--config-name", "experiment/pd_morl",
        *(f"{key}={value}" for key, value in settings.items()),
    ]
    if checkpoint is not None:
        command.append(f"+resume_from_checkpoint={checkpoint}")
    return command


def expected_counters(overrides=None):
    config = dict(total_timesteps=TOTAL, learning_start_timesteps=4160, num_envs=160,
                  rollout_length=8, fold_iters=4, critic_updates_per_rollout=640,
                  actor_update_interval=10)
    config.update(overrides or {})
    iterations, remainder = divmod(
        config["total_timesteps"] - config["learning_start_timesteps"],
        config["num_envs"] * config["rollout_length"],
    )
    if iterations < 0 or remainder or iterations % config["fold_iters"]:
        raise ValueError("budget must include prefill and whole host chunks")
    critic = iterations * config["critic_updates_per_rollout"]
    return dict(environment_step_count=config["total_timesteps"],
                inner_rollout_count=iterations, host_chunk_count=iterations // config["fold_iters"],
                critic_optimizer_step_count=critic,
                actor_optimizer_step_count=critic // config["actor_update_interval"],
                target_update_count=critic // config["actor_update_interval"])


def completed_result(run_dir, seed, overrides=None):
    """Check completion evidence, not a performance or convergence threshold."""
    counters = read_json(run_dir / "counters_summary.json")
    expected_counts = expected_counters(overrides)
    total = expected_counts["environment_step_count"]
    iterations = expected_counts["inner_rollout_count"]
    for key, expected in expected_counts.items():
        if counters.get(key) != expected:
            raise ValueError(f"{key}: {counters.get(key)} != {expected}")
    final = read_json(run_dir / "final_training_eval/results.json")
    if final["actual_env_transitions"] != total or final["iteration"] != iterations:
        raise ValueError("final evaluation has the wrong training budget")
    if len(final["preferences"]) != 1001 or len(final["returns_per_repeat"]) != 3:
        raise ValueError("final evaluation must have 1001 preferences x 3 repeats")
    for repeat in final["returns_per_repeat"]:
        if len(repeat) != 1001 or any(
            len(row) != 2 or not all(math.isfinite(x) for x in row) for row in repeat
        ):
            raise ValueError("incomplete or nonfinite final returns")
    metadata = read_json(run_dir / "run_metadata.json")
    if (metadata["training_seed"] != seed or metadata["jax_backend"] != "gpu"
            or metadata["visible_gpu_count"] != 1):
        raise ValueError("seed or single-GPU backend mismatch")
    if overrides:
        config = metadata["frozen_hyperparameters"]
        for key in ("total_timesteps", "rollout_length", "critic_updates_per_rollout"):
            if config[key] != overrides[key]:
                raise ValueError(f"completed run has wrong {key}")
        if metadata["key_objective_normalization"] != {"initial": "l2", "online": "l2"}:
            raise ValueError("sweep requires initial/online L2")
    if not (run_dir / f"checkpoints/{iterations}/default/_METADATA").is_file():
        raise ValueError("final checkpoint is missing")
    # A resume at the final checkpoint has no new learner diagnostics.
    diagnostics_path = run_dir / "training_diagnostics.jsonl"
    if diagnostics_path.exists():
        with diagnostics_path.open(encoding="utf-8") as handle:
            diagnostics = [json.loads(line) for line in handle if line.strip()]
    elif Path(metadata.get("resumed_from_checkpoint", "")).name == str(iterations):
        diagnostics = []
    else:
        raise ValueError("learner diagnostics are missing")
    if any(not row.get("finite", False) for row in diagnostics):
        raise ValueError("nonfinite learner diagnostics")
    history_path = run_dir / "hv_history.csv"
    if history_path.exists() or overrides:
        rollout_length = (overrides or {}).get("rollout_length", 8)
        chunk = 160 * rollout_length * 4
        expected_steps = [4160 + math.ceil((threshold - 4160) / chunk) * chunk
                          for threshold in range(250_000, total + 1, 250_000)]
        if expected_steps or history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        else:
            history = []
        if ([int(row["actual_env_transitions"]) for row in history] != expected_steps
                or any(int(row["preference_count"]) != 201 or int(row["repeats"]) != 3
                       for row in history)):
            raise ValueError("full HV history is incomplete or duplicated")
    else:
        # ponytail: read historical 51-point scans without rewriting their outputs.
        with (run_dir / "hv_convergence.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        if ([int(row["target_transition_threshold"]) for row in history]
                != list(range(250_000, total + 1, 250_000))):
            raise ValueError("convergence history is incomplete or duplicated")
    result = {key: final[key] for key in ("source_hv", "source_sparsity", "source_pareto_point_count")}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite final metrics")
    return result


def gpu_status():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    return {int(index): (int(free), int(util))
            for index, free, util in csv.reader(output.splitlines(), skipinitialspace=True)}


def alive_child(job):
    if job.get("status") != "running":
        return False
    try:
        command = Path(f"/proc/{job['pid']}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return f"hydra.run.dir={job['run_dir']}".encode() in command


class AdoptedProcess:
    """Watch a surviving training process after its controller is replaced."""

    def __init__(self, job):
        self.job = job
        self.pid = job["pid"]
        self.returncode = None

    def poll(self):
        if not alive_child(self.job):
            # Its exit status is unavailable to a new parent. Completion must
            # still pass the same full-budget/result/checkpoint checks below.
            self.returncode = 0
        return self.returncode

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else math.inf
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.1)
        return self.returncode


def next_ready_seed(pending, jobs, now):
    for _ in range(len(pending)):
        seed = pending.popleft()
        if jobs.get(str(seed), {}).get("retry_after", 0) <= now:
            return seed
        pending.append(seed)
    return None


def retry_job(job, pending, initial_delay, error):
    count = job.get("failures", 0) + 1
    delay = min(initial_delay * 2 ** min(count - 1, 10), 900)
    job.update(status="retry_wait", error=str(error), failures=count,
               retry_after=time.time() + delay)
    pending.append(job.get("job_id", job["seed"]))


def run_scan(args, root, seeds, plans=None):
    import fcntl  # Linux flock releases automatically if the controller exits.

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".scan.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another controller is using this output root") from error
        state_path = root / "scan.json"
        contract = ({"seeds": seeds, "total_timesteps": TOTAL} if plans is None
                    else {"plans": plans})
        contract["source_sha256"] = source_digest()
        state = read_json(state_path) if state_path.exists() else {"contract": contract, "jobs": {}}
        if state["contract"] != contract:
            raise ValueError("seeds/code/config changed; use a new output root")
        surviving = [job for job in state["jobs"].values() if alive_child(job)]
        max_jobs = getattr(args, "max_jobs_per_gpu", 1)
        occupied_slots = [(job["gpu"], job.get("slot", 0)) for job in surviving]
        if (len(set(occupied_slots)) != len(occupied_slots)
                or any(slot >= max_jobs for _, slot in occupied_slots)):
            raise RuntimeError("surviving children exceed GPU slots")
        if not {job["gpu"] for job in surviving} <= set(args.gpus):
            raise ValueError("include all GPUs with surviving children in --gpus")
        for job in state["jobs"].values():
            if job["status"] == "complete":
                if plans is None:
                    completed_result(Path(job["run_dir"]), job["seed"])
                else:
                    completed_result(Path(job["run_dir"]), job["seed"], job["overrides"])
        live_seeds = {job.get("job_id", job["seed"]) for job in surviving}
        pending = deque(sorted(
            (seed for seed in seeds if seed not in live_seeds
             and state["jobs"].get(str(seed), {}).get("status") != "complete"),
            key=lambda seed: (str(seed) not in state["jobs"], seeds.index(seed)),
        ))
        devices = gpu_status()
        if not set(args.gpus) <= devices.keys():
            raise ValueError(f"requested GPUs not found: {args.gpus}; available: {list(devices)}")
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < len(args.gpus) * max_jobs * args.cpus_per_job:
            raise ValueError("not enough allowed CPUs; lower --cpus-per-job")
        cache = REPO / "jax_cache/pd_morl_seed_scan" / hashlib.sha256(str(root).encode()).hexdigest()[:12]
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        active = {}
        for job in surviving:
            job["adopted"] = True
            active[(job["gpu"], job.get("slot", 0))] = (AdoptedProcess(job), None, job)
            print(f"GPU {job['gpu']}: adopted seed {job['seed']} PID {job['pid']}", flush=True)
        failures = 0
        stop = False

        def request_stop(signum, frame):
            nonlocal stop
            stop = True

        previous_handlers = {sig: signal.signal(sig, request_stop)
                             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        last_status = 0.0
        write_json(state_path, state)
        try:
            while (pending or active) and not stop:
                try:
                    devices = gpu_status()
                except (OSError, subprocess.SubprocessError) as error:
                    print(f"GPU status unavailable; keeping active jobs: {error}", flush=True)
                    time.sleep(args.poll_seconds)
                    continue
                for gpu_slot, (process, log, job) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    if log is not None:
                        log.close()
                    job.update(returncode=None if job.get("adopted") else code,
                               finished_utc=datetime.now(timezone.utc).isoformat())
                    try:
                        if code:
                            raise RuntimeError(f"training exited with code {code}")
                        if plans is None:
                            job["result"] = completed_result(Path(job["run_dir"]), job["seed"])
                        else:
                            job["result"] = completed_result(Path(job["run_dir"]), job["seed"], job["overrides"])
                        job["status"] = "complete"
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        retry_job(job, pending, args.retry_delay_seconds, error)
                        failures += 1
                    print(f"GPU {gpu_slot[0]}: {job.get('job_id', job['seed'])} {job['status']} {job.get('error', '')}", flush=True)
                    del active[gpu_slot]
                    write_json(state_path, state)
                if pending:
                    reserved_slots = getattr(args, "reserved_slots", set)()
                    reserved = {gpu: 0 for gpu in args.gpus}
                    # Check both slots; memory reserved by this pass is not yet visible to nvidia-smi.
                    for gpu, slot in ((gpu, slot) for slot in range(max_jobs)
                                      for gpu in sorted(args.gpus, key=lambda g: devices[g][1])):
                        if not pending or stop:
                            break
                        if ((gpu, slot) in active or (gpu, slot) in reserved_slots or
                                devices[gpu][0] - reserved[gpu] < args.min_free_gb * 1024):
                            continue
                        # Reject source/config drift before starting another seed.
                        if source_digest() != contract["source_sha256"]:
                            raise RuntimeError("source/config changed during the scan")
                        seed = next_ready_seed(pending, state["jobs"], time.time())
                        if seed is None:
                            break
                        plan = plans[seed] if plans is not None else {"seed": seed}
                        seed_dir = root / (seed if plans is not None else f"seed_{seed:04d}")
                        attempts = [int(p.name.split("_")[1]) for p in seed_dir.glob("attempt_*") if p.is_dir()]
                        run_dir = seed_dir / f"attempt_{max(attempts, default=0) + 1:03d}"
                        checkpoint = latest_checkpoint(seed_dir) or plan.get("resume_from_checkpoint")
                        command = training_command(plan["seed"], run_dir, checkpoint, plan.get("overrides"))
                        offset = (args.gpus.index(gpu) * max_jobs + slot) * args.cpus_per_job
                        affinity = cpus[offset:offset + args.cpus_per_job]
                        command = ["taskset", "-c", ",".join(map(str, affinity)), *command]
                        env = os.environ.copy()
                        env.pop("JAX_PLATFORMS", None)
                        gpu_cache = cache / f"gpu_{gpu}"
                        if max_jobs > 1:
                            gpu_cache = gpu_cache / seed_dir.name / run_dir.name
                        gpu_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
                        env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                                   XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_COMPILATION_CACHE_DIR=str(gpu_cache),
                                   JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="none",
                                   OMP_NUM_THREADS=str(args.cpus_per_job), OPENBLAS_NUM_THREADS="1",
                                   PYTHONUNBUFFERED="1", HYDRA_FULL_ERROR="1")
                        run_dir.mkdir(parents=True, exist_ok=False)
                        job = dict(seed=plan["seed"], job_id=seed, gpu=gpu, slot=slot,
                                   run_dir=str(run_dir), status="starting",
                                   failures=state["jobs"].get(str(seed), {}).get("failures", 0),
                                   resume_from=str(checkpoint) if checkpoint else None, command=command,
                                   started_utc=datetime.now(timezone.utc).isoformat())
                        if plans is not None:
                            job["overrides"] = plan["overrides"]
                        state["jobs"][str(seed)] = job
                        log = (run_dir / "launcher.log").open("w", encoding="utf-8")
                        try:
                            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                        except OSError as error:
                            log.close()
                            retry_job(job, pending, args.retry_delay_seconds, error)
                            failures += 1
                            write_json(state_path, state)
                            print(f"GPU {gpu}: seed {seed} launch failed, will retry: {error}", flush=True)
                            continue
                        except BaseException:
                            log.close()
                            raise
                        job.update(pid=process.pid, status="running")
                        active[(gpu, slot)] = (process, log, job)
                        reserved[gpu] += args.min_free_gb * 1024
                        write_json(state_path, state)
                        print(f"GPU {gpu}: seed {seed} started, resume={checkpoint}, output={run_dir}", flush=True)
                if time.monotonic() - last_status >= 60:
                    done = sum(job["status"] == "complete" for job in state["jobs"].values())
                    print(f"complete={done}/{len(seeds)} running={len(active)} pending={len(pending)} "
                          f"failures_this_invocation={failures}; free MiB/util%={devices}", flush=True)
                    last_status = time.monotonic()
                if pending or active:
                    time.sleep(args.poll_seconds)
        finally:
            # Only signal this scan's created or explicitly adopted children.
            for process, _, _ in active.values():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process, log, job in active.values():
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if log is not None:
                    log.close()
                job.update(status="interrupted", returncode=process.returncode)
            write_json(state_path, state)
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        done = sum(job["status"] == "complete" for job in state["jobs"].values())
        print(f"Finished: {done}/{len(seeds)} complete. State: {state_path}", flush=True)
        return 0 if done == len(seeds) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=7)
    parser.add_argument("--seed-count", type=int, default=80)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/pd_morl_80seeds_step7_10m_l8_c640"))
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    parser.add_argument("--cpus-per-job", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.seed_start < 0 or min(args.seed_step, args.seed_count, args.cpus_per_job) < 1
            or not math.isfinite(args.retry_delay_seconds) or args.retry_delay_seconds <= 0
            or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0
            or not math.isfinite(args.min_free_gb) or args.min_free_gb < 0
            or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0):
        parser.error("require positive counts/step/polling, nonnegative seeds/memory, and distinct GPU indices")
    seeds = [args.seed_start + args.seed_step * i for i in range(args.seed_count)]
    if seeds[-1] >= 2**32:
        parser.error("seeds must fit uint32")
    root = (REPO / args.output_root).resolve()
    print(f"Seeds ({len(seeds)}): {seeds}\nTotal base transitions: {len(seeds) * TOTAL:,}", flush=True)
    if args.dry_run:
        print(f"Dynamic GPU queue: {args.gpus}; one job/GPU; minimum free memory {args.min_free_gb} GiB")
        print("Example: " + shlex.join(training_command(seeds[0], root / f"seed_{seeds[0]:04d}/attempt_001")))
        return 0
    if sys.platform != "linux":
        parser.error("execute training on Linux/lab4090; --dry-run also works locally")
    return run_scan(args, root, seeds)


if __name__ == "__main__":
    raise SystemExit(main())
