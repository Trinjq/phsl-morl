"""Run key seeds 0/1/2, then 3/4/5, with one preference per GPU."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts/train_brax_key_solutions.py"
PREFERENCES = [[0., 1.], [.5, .5], [1., 0.]]


def run(output_dir, smoke=False, dry_run=False):
    state = {"status": "planned", "smoke": smoke,
             "started_utc": datetime.now(timezone.utc).isoformat(), "groups": []}
    for group_index, seeds in enumerate([[0, 1, 2], [3, 4, 5]]):
        group_dir = output_dir / f"group_{group_index + 1}"
        jobs = [{"seed": seed, "key_index": gpu, "gpu": gpu,
                 "preference": PREFERENCES[gpu], "status": "queued",
                 "output_dir": str(group_dir / f"key_{gpu}_seed_{seed}")}
                for gpu, seed in enumerate(seeds)]
        state["groups"].append({"seeds": seeds, "output_dir": str(group_dir), "jobs": jobs})
    assert [j["seed"] for g in state["groups"] for j in g["jobs"]] == list(range(6))
    if dry_run:
        print(json.dumps(state, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=False)
    state["training_script_sha256"] = hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    state["status"] = "running"

    def save():
        temporary = output_dir / "run.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output_dir / "run.json")

    def stop(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 24:
        raise RuntimeError("three GPU workers need at least 24 available CPU cores")
    active = []
    save()
    try:
        for group in state["groups"]:
            group_dir = Path(group["output_dir"])
            group_dir.mkdir()
            for job in group["jobs"]:
                gpu = job["gpu"]
                env = os.environ.copy()
                env.pop("JAX_PLATFORMS", None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
                           JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="none",
                           JAX_COMPILATION_CACHE_DIR=str(output_dir / "jax_cache" / f"gpu_{gpu}"),
                           OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
                command = ["taskset", "-c", ",".join(map(str, cpus[8*gpu:8*gpu+8])),
                           sys.executable, str(TRAIN), "--key-index", str(job["key_index"]),
                           "--seed", str(job["seed"]), "--output-dir", job["output_dir"]]
                if smoke:
                    command.append("--smoke")
                log = (group_dir / f"seed_{job['seed']}.log").open("w")
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status="running", pid=process.pid,
                           started_utc=datetime.now(timezone.utc).isoformat())
                active.append((process, log, job))
                save()
                print(f"GPU {gpu}: seed {job['seed']} started, PID {process.pid}", flush=True)
            while active:
                for process, log, job in list(active):
                    code = process.poll()
                    if code is None:
                        continue
                    job.update(status="complete" if code == 0 else "failed", returncode=code,
                               finished_utc=datetime.now(timezone.utc).isoformat())
                    log.close()
                    active.remove((process, log, job))
                    save()
                    if code:
                        raise RuntimeError(f"seed {job['seed']} exited with code {code}; inspect its log")
                    print(f"seed {job['seed']} complete", flush=True)
                if active:
                    time.sleep(3)
            env = os.environ.copy()
            env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1")
            subprocess.run([sys.executable, str(TRAIN), "--output-dir", str(group_dir),
                            "--assemble-seeds", *map(str, group["seeds"])],
                           cwd=REPO, env=env, check=True)
            group["artifact"] = str(group_dir / "interp_objs_walker2d_brax.candidate.txt")
            save()
        state["status"] = "complete"
    except KeyboardInterrupt:
        state["status"] = "stopped"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        for process, _, _ in active:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process, log, job in active:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            log.close()
            job.update(status="interrupted", returncode=process.returncode)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.smoke, args.dry_run)
