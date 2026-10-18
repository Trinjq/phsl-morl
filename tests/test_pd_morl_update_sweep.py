"""Check the real sweep queue with fake processes; no training is launched."""

import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_pd_morl_update_sweep as sweep

scan = sweep.scan


def test_sampling_cadence_keeps_total_updates_and_reuses_baseline(tmp_path, capsys):
    plans = sweep.make_sampling_plans([0, 1, 2], [2, 8], 2_000_960)
    assert len(plans) == 6
    for name, plan in plans.items():
        length = plan["overrides"]["rollout_length"]
        assert length in (2, 8)
        assert plan["overrides"]["critic_updates_per_rollout"] == 80 * length
        counts = scan.expected_counters(plan["overrides"])
        assert counts["critic_optimizer_step_count"] == 998400
        assert counts["actor_optimizer_step_count"] == 99840
        assert counts["environment_step_count"] == 2000960
    assert (
        sweep.main(
            [
                "--sampling-lengths",
                "2",
                "8",
                "--dry-run",
                "--output-root",
                str(tmp_path / "preview"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Jobs: 6" in output and "max jobs/GPU: 2" in output
    assert "rollout_length=4" not in output
    assert not (tmp_path / "preview").exists()
    for lengths in (["2", "2"], ["3"]):
        with pytest.raises(SystemExit):
            sweep.main(
                [
                    "--sampling-lengths",
                    *lengths,
                    "--dry-run",
                    "--output-root",
                    str(tmp_path),
                ]
            )


def test_default_plan_and_completion(tmp_path, capsys):
    assert sweep.main(["--dry-run", "--output-root", str(tmp_path / "preview")]) == 0
    output = capsys.readouterr().out
    assert "Jobs: 9" in output and "max jobs/GPU: 2" in output
    assert not (tmp_path / "preview").exists()
    plans = sweep.make_plans([0, 1, 2], [80, 160, 320], 2_000_960)
    assert len(plans) == 9
    for critic in (80, 160, 320):
        plan = plans[f"critic_{critic}_seed_0"]
        overrides = plan["overrides"]
        expected = scan.expected_counters(overrides)
        assert expected["inner_rollout_count"] == 3120
        assert expected["host_chunk_count"] == 780
        assert expected["critic_optimizer_step_count"] == critic * 3120
        assert expected["actor_optimizer_step_count"] == critic * 312
        run = tmp_path / f"c{critic}"
        (run / "final_training_eval").mkdir(parents=True)
        checkpoint = run / "checkpoints/3120/default"
        checkpoint.mkdir(parents=True)
        scan.write_json(checkpoint / "_METADATA", {})
        scan.write_json(run / "counters_summary.json", expected)
        scan.write_json(
            run / "run_metadata.json",
            {
                "training_seed": 0,
                "jax_backend": "gpu",
                "visible_gpu_count": 1,
                "key_objective_normalization": {"initial": "l2", "online": "l2"},
                "frozen_hyperparameters": overrides,
            },
        )
        scan.write_json(
            run / "final_training_eval/results.json",
            {
                "actual_env_transitions": 2_000_960,
                "iteration": 3120,
                "preferences": [[0.0, 1.0]] * 1001,
                "returns_per_repeat": [[[1.0, 2.0]] * 1001] * 3,
                "source_hv": 2.0,
                "source_sparsity": 0.0,
                "source_pareto_point_count": 1,
            },
        )
        (run / "training_diagnostics.jsonl").write_text('{"finite": true}\n')
        with (run / "hv_history.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["actual_env_transitions", "preference_count", "repeats"])
            writer.writerows(
                [4160 + ((threshold - 4160 + 2559) // 2560) * 2560, 201, 3]
                for threshold in range(250_000, 2_000_001, 250_000)
            )
        assert scan.completed_result(run, 0, overrides)["source_hv"] == 2.0
        wrong = dict(overrides, critic_updates_per_rollout=critic + 10)
        with pytest.raises(ValueError, match="critic_optimizer_step_count"):
            scan.completed_result(run, 0, wrong)
        with (run / "hv_history.csv").open("a") as handle:
            handle.write("2000960,201,3\n")
        with pytest.raises(ValueError, match="HV history"):
            scan.completed_result(run, 0, overrides)
    assert scan.expected_counters() == scan.EXPECTED
    for arguments in (
        ["--total-timesteps", "2000000"],
        ["--seeds", "0", "0"],
        ["--min-free-gb", "nan"],
        ["--max-jobs-per-gpu", "3"],
    ):
        with pytest.raises(SystemExit):
            sweep.main(["--dry-run", "--output-root", str(tmp_path), *arguments])


@pytest.mark.skipif(sys.platform != "linux", reason="Linux flock and CPU affinity")
def test_sweep_two_slots_memory_guard_retry_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "REPO", tmp_path)
    monkeypatch.setattr(scan, "source_digest", lambda: "same-training")
    monkeypatch.setattr(scan.os, "sched_getaffinity", lambda _: set(range(24)))
    monkeypatch.setattr(scan.time, "sleep", lambda _: None)
    free = {0: 25_000, 1: 25_000, 2: 12_000}
    monkeypatch.setattr(
        scan, "gpu_status", lambda: {g: (f, 0) for g, f in free.items()}
    )
    plans = sweep.make_plans([0, 1, 2], [80, 160, 320], 2_000_960)
    counts, peaks = {g: 0 for g in free}, {g: 0 for g in free}
    for name, plan in plans.items():
        plan["resume_from_checkpoint"] = f"/previous/{name}/checkpoints/780"
    caches, affinity, launches = set(), set(), []
    failed = set()
    reserved_slot = [True]

    class Child:
        def __init__(self, command, **kwargs):
            self.gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            self.name = next(
                x.split("=", 1)[1] for x in command if x.startswith("+run_id=")
            )
            self.run_dir = Path(
                next(
                    x.split("=", 1)[1]
                    for x in command
                    if x.startswith("hydra.run.dir=")
                )
            )
            checkpoints = [x for x in command if x.startswith("+resume_from_checkpoint=")]
            expected = (self.run_dir.parent / "attempt_001/checkpoints/1564"
                        if self.run_dir.name == "attempt_002"
                        else plans[self.name]["resume_from_checkpoint"])
            assert checkpoints == [f"+resume_from_checkpoint={expected}"]
            counts[self.gpu] += 1
            peaks[self.gpu] = max(peaks[self.gpu], counts[self.gpu])
            assert counts[self.gpu] <= 2
            assert sum(counts.values()) <= 6
            self.cache = kwargs["env"]["JAX_COMPILATION_CACHE_DIR"]
            assert self.cache not in caches
            caches.add(self.cache)
            self.cpus = set(command[2].split(","))
            slot = (min(map(int, self.cpus)) // 4) % 2
            assert not (self.gpu == 1 and slot == 0 and reserved_slot[0])
            assert not affinity & self.cpus
            affinity.update(self.cpus)
            assert kwargs["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
            self.pid, self.returncode, self.remaining = 900000 + len(launches), None, 3
            launches.append(command)

        def poll(self):
            if len(launches) == 4:
                assert (
                    counts[2] == 1
                )  # The first pass reserved memory for one GPU-2 job.
                assert counts[1] == 1  # Another queue occupies GPU-1 slot 0.
                free[2] = 25_000
                reserved_slot[0] = False
            if self.returncode is None:
                self.remaining -= 1
                if self.remaining <= 0:
                    counts[self.gpu] -= 1
                    affinity.difference_update(self.cpus)
                    free[2] = 25_000
                    self.returncode = (
                        1 if self.name == "critic_80_seed_0" and not failed else 0
                    )
                    if self.returncode:
                        failed.add(self.name)
                        checkpoint = self.run_dir / "checkpoints/1564/default"
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "_METADATA").write_text("{}")
            return self.returncode

    monkeypatch.setattr(scan.subprocess, "Popen", Child)
    monkeypatch.setattr(
        scan, "completed_result", lambda run, seed, overrides: {"source_hv": seed + 1}
    )
    args = SimpleNamespace(
        gpus=[0, 1, 2],
        max_jobs_per_gpu=2,
        cpus_per_job=4,
        min_free_gb=8,
        poll_seconds=0.001,
        retry_delay_seconds=0.001,
        reserved_slots=lambda: {(1, 0)} if reserved_slot[0] else set(),
    )
    root = tmp_path / "scan"
    assert scan.run_scan(args, root, list(plans), plans) == 0
    assert peaks == {0: 2, 1: 2, 2: 2}
    assert len(launches) == 10 and sum(counts.values()) == 0
    assert any(
        any(x.endswith("attempt_001/checkpoints/1564") for x in cmd) for cmd in launches
    )
    state = scan.read_json(root / "scan.json")
    assert len(state["jobs"]) == 9 and all(
        j["status"] == "complete" for j in state["jobs"].values()
    )
    assert scan.run_scan(args, root, list(plans), plans) == 0
    assert len(launches) == 10
    changed = sweep.make_plans([0, 1, 2], [80, 160, 320], 10_000_960)
    with pytest.raises(ValueError, match="changed"):
        scan.run_scan(args, root, list(changed), changed)
