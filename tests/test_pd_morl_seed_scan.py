"""Exercise the real queue with fake children; never launch a training job."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_pd_morl_seed_scan as scan


def test_default_seed_plan_and_dry_run_have_no_side_effects(tmp_path, capsys):
    root = tmp_path / "not-created"
    assert scan.main(["--dry-run", "--output-root", str(root)]) == 0
    output = capsys.readouterr().out
    assert "Seeds (80): [0, 7, 14," in output and "546, 553]" in output
    assert "800,281,600" in output and "experiment/pd_morl" in output
    assert "+run_id=pd_morl_seed_0" in output
    assert not root.exists()


def test_default_training_config_matches_queue_budget_and_checkpoint_units():
    from omegaconf import OmegaConf

    config = OmegaConf.load(scan.REPO / "configs/experiment/pd_morl.yaml")
    assert (config.rollout_length, config.critic_updates_per_rollout,
            config.replay_batch_size, config.actor_update_interval) == (8, 640, 512, 10)
    keys = ("total_timesteps", "learning_start_timesteps", "num_envs", "rollout_length",
            "fold_iters", "critic_updates_per_rollout", "actor_update_interval")
    assert scan.expected_counters({key: config[key] for key in keys}) == scan.EXPECTED
    assert config.total_timesteps == scan.TOTAL
    interval = config.checkpoint.save_interval_steps
    assert interval % config.fold_iters == 0
    assert 1_000_000 <= interval * config.num_envs * config.rollout_length < 1_010_000


def test_checkpoint_selection_ignores_incomplete_and_temporary_saves(tmp_path):
    for attempt, step in [(1, "1564"), (1, "3128"), (2, "4692"), (2, "6256.orbax-checkpoint-tmp")]:
        path = tmp_path / f"attempt_{attempt:03d}/checkpoints/{step}/default"
        path.mkdir(parents=True)
        (path / "_METADATA").write_text("{}")
    (tmp_path / "attempt_002/checkpoints/6256").mkdir()
    selected = scan.latest_checkpoint(tmp_path)
    assert selected.name == "4692"
    assert scan.training_command(7, tmp_path, selected)[-1] == f"+resume_from_checkpoint={selected}"


@pytest.mark.skipif(sys.platform != "linux", reason="lab4090 uses Linux flock and CPU affinity")
def test_dynamic_queue_resume_failure_and_no_gpu_collision(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "REPO", tmp_path)
    monkeypatch.setattr(scan, "source_digest", lambda: "same-code-config")
    monkeypatch.setattr(scan.os, "sched_getaffinity", lambda _: set(range(24)))
    now = [1000.0]
    monkeypatch.setattr(scan.time, "time", lambda: now[0])
    def advance(_):
        now[0] += 1
    monkeypatch.setattr(scan.time, "sleep", advance)
    launched = []

    def devices():
        # GPU 2 starts below the memory threshold, then becomes available.
        return {0: (12000, 20), 1: (12000, 30), 2: (3000 if len(launched) < 3 else 12000, 10)}

    monkeypatch.setattr(scan, "gpu_status", devices)
    occupied = set()
    fail = {7}

    class Child:
        def __init__(self, command, **kwargs):
            self.gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            assert self.gpu not in occupied, "two seeds assigned to the same GPU"
            occupied.add(self.gpu)
            self.seed = int(next(x for x in command if x.startswith("seed=")).split("=")[1])
            self.run_dir = Path(next(x for x in command if x.startswith("hydra.run.dir=")).split("=", 1)[1])
            self.pid = 900000 + len(launched)
            self.returncode = None
            self.remaining = 5 if self.seed == 0 else 1
            launched.append((self.gpu, self.seed, command))
            assert kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
            assert "JAX_COMPILATION_CACHE_DIR" in kwargs["env"]
            assert kwargs["env"]["JAX_COMPILATION_CACHE_DIR"].endswith(f"gpu_{self.gpu}")
            assert kwargs["env"]["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] == "none"
            assert kwargs["start_new_session"]

        def poll(self):
            if self.returncode is None:
                self.remaining -= 1
                if self.remaining <= 0:
                    self.returncode = 1 if self.seed in fail else 0
                    fail.discard(self.seed)
                    occupied.remove(self.gpu)
                    if self.returncode:
                        checkpoint = self.run_dir / "checkpoints/1564/default"
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "_METADATA").write_text("{}")
            return self.returncode

    monkeypatch.setattr(scan.subprocess, "Popen", Child)
    monkeypatch.setattr(scan, "completed_result", lambda path, seed: {"source_hv": seed + 1})
    args = SimpleNamespace(gpus=[0, 1, 2], cpus_per_job=8, retry_delay_seconds=1,
                           min_free_gb=8, poll_seconds=0.001)
    root = tmp_path / "outputs/scan"
    seeds = [0, 7, 14, 21, 28, 35]
    assert scan.run_scan(args, root, seeds) == 0
    assert not occupied and len(launched) == len(seeds) + 1
    assert {gpu for gpu, _, _ in launched} == {0, 1, 2}
    assert next(g for g, s, _ in launched if s == 14) == 1  # fast GPU takes the next seed
    state = json.loads((root / "scan.json").read_text())
    assert state["jobs"]["7"]["status"] == "complete"
    assert state["jobs"]["7"]["failures"] == 1
    assert scan.run_scan(args, root, seeds) == 0
    assert len(launched) == len(seeds) + 1  # completed seeds were skipped
    assert launched[-1][1] == 7
    assert any(x.endswith("attempt_001/checkpoints/1564") for x in launched[-1][2])
    assert (root / "seed_0007/attempt_001/launcher.log").exists()
    assert (root / "seed_0007/attempt_002/launcher.log").exists()
    monkeypatch.setattr(scan, "source_digest", lambda: "changed-code-config")
    with pytest.raises(ValueError, match="changed"):
        scan.run_scan(args, root, seeds)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux controller takeover")
def test_takeover_keeps_live_seed_and_retries_previously_failed_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "REPO", tmp_path)
    monkeypatch.setattr(scan, "source_digest", lambda: "same-training")
    monkeypatch.setattr(scan.os, "sched_getaffinity", lambda _: {0, 1})
    monkeypatch.setattr(scan, "gpu_status", lambda: {0: (12000, 10), 1: (12000, 20)})
    monkeypatch.setattr(scan.time, "sleep", lambda _: None)
    alive = [True]
    monkeypatch.setattr(scan, "alive_child", lambda job: job.get("pid") == 987654 and alive[0])
    monkeypatch.setattr(scan, "completed_result", lambda path, seed: {"source_hv": 1})
    root = tmp_path / "scan"
    root.mkdir()
    scan.write_json(root / "scan.json", {
        "contract": {"seeds": [0, 7], "total_timesteps": scan.TOTAL, "source_sha256": "same-training"},
        "jobs": {"0": {"seed": 0, "gpu": 0, "pid": 987654, "status": "running", "run_dir": "/existing/seed0"},
                 "7": {"seed": 7, "gpu": 1, "pid": 987655, "status": "failed", "run_dir": "/old/seed7"}},
    })
    launched = []
    class Child:
        pid = 987656
        returncode = None
        def __init__(self, command, **kwargs):
            assert "seed=7" in command and "seed=0" not in command
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
            launched.append(command)
        def poll(self):
            alive[0] = False
            self.returncode = 0
            return 0
    monkeypatch.setattr(scan.subprocess, "Popen", Child)
    args = SimpleNamespace(gpus=[0, 1], cpus_per_job=1, retry_delay_seconds=1,
                           min_free_gb=8, poll_seconds=0.001)
    assert scan.run_scan(args, root, [0, 7]) == 0
    state = scan.read_json(root / "scan.json")
    assert len(launched) == 1
    assert state["jobs"]["0"]["pid"] == 987654
    assert state["jobs"]["0"]["adopted"]
    assert state["jobs"]["0"]["returncode"] is None
    assert all(job["status"] == "complete" for job in state["jobs"].values())


def test_retry_backoff_does_not_block_other_seeds():
    from collections import deque
    pending = deque([7, 14])
    jobs = {"7": {"retry_after": 100}}
    assert scan.next_ready_seed(pending, jobs, 50) == 14
    assert scan.next_ready_seed(pending, jobs, 99) is None
    assert list(pending) == [7]
    assert scan.next_ready_seed(pending, jobs, 100) == 7
    job = {"seed": 7, "failures": 10}
    before = scan.time.time()
    scan.retry_job(job, pending, 60, OSError("spawn temporarily unavailable"))
    assert job["status"] == "retry_wait" and job["failures"] == 11
    assert before + 900 <= job["retry_after"] <= scan.time.time() + 900


def test_completion_requires_final_evaluation_and_checkpoint(tmp_path):
    def write(name, payload):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    write("counters_summary.json", scan.EXPECTED)
    write("final_training_eval/results.json", {
        "actual_env_transitions": scan.TOTAL, "iteration": scan.ITERATIONS,
        "preferences": [[0, 1]] * 1001, "returns_per_repeat": [[[1, 2]] * 1001] * 3,
        "source_hv": 2, "source_sparsity": 0, "source_pareto_point_count": 1,
    })
    write("run_metadata.json", {"training_seed": 7, "jax_backend": "gpu", "visible_gpu_count": 1})
    write("training_diagnostics.jsonl", {"finite": True})
    (tmp_path / "hv_convergence.csv").write_text(
        "target_transition_threshold\n" + "\n".join(map(str, range(250000, 10000001, 250000))) + "\n")
    with pytest.raises(ValueError, match="checkpoint"):
        scan.completed_result(tmp_path, 7)
    write(f"checkpoints/{scan.ITERATIONS}/default/_METADATA", {})
    assert scan.completed_result(tmp_path, 7)["source_hv"] == 2
    # Final-checkpoint resume only needs to finish evaluation, with no new updates.
    (tmp_path / "training_diagnostics.jsonl").unlink()
    write("run_metadata.json", {"training_seed": 7, "jax_backend": "gpu", "visible_gpu_count": 1,
                                "resumed_from_checkpoint": f"/previous/checkpoints/{scan.ITERATIONS}"})
    assert scan.completed_result(tmp_path, 7)["source_hv"] == 2
    (tmp_path / "final_training_eval/results.json").unlink()
    with pytest.raises(FileNotFoundError):
        scan.completed_result(tmp_path, 7)
