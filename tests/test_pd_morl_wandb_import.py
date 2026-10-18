import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from upload_pd_morl_records_wandb import (
    completed_jobs,
    record_digest,
    scalar_metrics,
    training_events,
)


def test_partial_import_only_reads_completed_jobs_and_later_includes_the_last_seed():
    state = {
        "contract": {"plans": {"seed0": {}, "seed60": {}}},
        "jobs": {"seed0": {"status": "complete"}, "seed60": {"status": "running"}},
    }
    with pytest.raises(ValueError, match="all planned"):
        completed_jobs(state)
    assert set(completed_jobs(state, True)) == {"seed0"}
    state["jobs"]["seed60"]["status"] = "complete"
    assert set(completed_jobs(state)) == {"seed0", "seed60"}


def test_import_merges_actual_steps_and_keeps_final_evaluation_separate(tmp_path):
    original = {
        "iteration": 100,
        "base_inserts": 68160,
        "critic_loss": 2.5,
        "critic_angle": [0.1, 0.2],
        "q_target": [[1.0, 2.0], [3.0, 4.0]],
    }
    (tmp_path / "training_diagnostics.jsonl").write_text(json.dumps(original) + "\n")
    with (tmp_path / "hv_history.csv").open("w", newline="") as handle:
        row = dict(
            actual_env_transitions=68160,
            source_hv=10.0,
            mean_repeat_hv=11.0,
            source_sparsity=2.0,
            source_pareto_point_count=3,
            preference_count=201,
            repeats=3,
            wall_clock_seconds=5.0,
        )
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    with (tmp_path / "key_replacement_diagnostics.csv").open("w", newline="") as handle:
        row = dict(
            actual_env_transitions=101440,
            key_index=0,
            candidate_r0=1.0,
            candidate_r1=2.0,
            stored_score=3.0,
            candidate_score=2.0,
            score_gap=-1.0,
            stored_r0_after=4.0,
            stored_r1_after=3.0,
            replaced="False",
        )
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    final = dict(
        actual_env_transitions=2000960,
        source_hv=20.0,
        source_sparsity=1.0,
        source_pareto_point_count=4,
        mean_repeat_hv=21.0,
        mean_repeat_sparsity=2.0,
        hv_per_repeat=[19.0, 20.0, 24.0],
        sparsity_per_repeat=[1.0, 2.0, 3.0],
        preferences=[[0.0, 1.0]] * 1001,
        returns_per_repeat=[[], [], []],
    )
    (tmp_path / "final_training_eval").mkdir()
    (tmp_path / "final_training_eval/results.json").write_text(json.dumps(final))
    files = list(tmp_path.rglob("*.json"))
    before = record_digest(tmp_path, files)
    events, _ = training_events(tmp_path)
    assert [step for step, _ in events] == [68160, 101440, 2000960]
    assert events[0][1]["train/critic_loss"] == 2.5
    assert events[0][1]["eval/source_hv"] == 10.0
    assert events[0][1]["eval/preference_count"] == 201
    assert events[1][1]["key/0/replaced"] == 0
    assert events[-1][1]["final/source_hv"] == 20.0
    assert events[-1][1]["final/preference_count"] == 1001
    assert "eval/source_hv" not in events[-1][1]
    assert "train/q_target" not in events[0][1]
    assert events[0][1]["train/critic_angle_1"] == 0.2
    assert record_digest(tmp_path, files) == before
    assert scalar_metrics("test", {"nan": float("nan"), "inf": float("inf")}) == {}


def test_sampling_workspace_reuses_only_the_three_matching_baselines(
    tmp_path, monkeypatch
):
    ws = pytest.importorskip("wandb_workspaces.workspaces")
    import wandb
    import create_pd_morl_wandb_workspace as charts

    def run(length, seed):
        return SimpleNamespace(
            id=f"l{length}s{seed}",
            state="finished",
            summary={"import_complete": True},
            config=dict(
                rollout_length=length,
                seed=seed,
                num_envs=160,
                replay_batch_size=512,
                actor_update_interval=10,
                total_timesteps=2000960,
                critic_updates_per_rollout=80 * length,
                key_artifact_sha256="same-v3",
            ),
        )

    new_runs = [run(length, seed) for length in (2, 8) for seed in (0, 1, 2)]
    baseline = [run(4, seed) for seed in (0, 1, 2)]
    full_seeds = [run(8, seed) for seed in range(0, 61, 10)]
    for item in full_seeds:
        item.config["total_timesteps"] = 10003520

    def get_runs(path, filters):
        if filters["group"] == "full-seeds":
            return full_seeds
        if filters["group"] == "sampling":
            return new_runs
        assert filters == {
            "group": "critic",
            "config.rollout_length": 4,
            "config.critic_updates_per_rollout": 320,
        }
        return baseline

    monkeypatch.setattr(wandb, "Api", lambda **kwargs: SimpleNamespace(runs=get_runs))
    saved = []

    class Workspace:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.url = "https://wandb.ai/entity/project?nw=test"

        def save(self):
            saved.append(self)

        @classmethod
        def from_url(cls, url):
            assert url == saved[-1].url
            return saved[-1]

    monkeypatch.setattr(ws, "Workspace", Workspace)
    monkeypatch.setattr(charts, "__file__", str(tmp_path / "scripts/charts.py"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["charts", "sampling", "--baseline-group", "critic", "--entity", "entity"],
    )
    charts.main()
    workspace = saved[-1]
    assert sum(len(s.panels) for s in workspace.sections) == 14
    assert set(workspace.runset_settings.run_settings) == {
        r.id for r in new_runs + baseline
    }
    assert "sampling" in str(workspace.runset_settings.filters)
    assert "critic" in str(workspace.runset_settings.filters)
    assert "rollout_length" in str(workspace.sections[0].panels[1].groupby)
    new_runs[0].config["critic_updates_per_rollout"] = 320
    with pytest.raises(ValueError, match="equal update/transition"):
        charts.main()
    assert len(saved) == 1
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "charts",
            "full-seeds",
            "--entity",
            "entity",
            "--seeds",
            *map(str, range(0, 61, 10)),
        ],
    )
    charts.main()
    workspace = saved[-1]
    assert len(workspace.runset_settings.run_settings) == 7
    assert workspace.sections[0].panels[0].max_runs_to_show == 7
    assert workspace.sections[0].panels[0].range_x == (0, 10003520)
    assert "critic_updates_per_rollout" in str(workspace.sections[0].panels[1].groupby)
    assert "seed" in str(workspace.sections[-1].panels[0].groupby)
    full_seeds[-1].state = "running"
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--allow-partial"])
    charts.main()
    assert len(saved[-1].runset_settings.run_settings) == 6
    assert "finished" in str(saved[-1].runset_settings.filters)
    pointer = tmp_path / "outputs/wandb_imports/full-seeds/visualization_workspace.json"
    assert json.loads(pointer.read_text())["pending_seeds"] == [60]
