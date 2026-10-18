import csv
import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from evorl.algorithms.mo_td3 import MOTD3Workflow, PDMORLGPUWorkflow
from evorl.types import PyTreeDict, State
from evorl.utils.pd_morl_convergence import (
    append_convergence_result,
    plot_convergence_csv,
    prepare_convergence_history,
)
from hydra import compose, initialize
from omegaconf import OmegaConf


def _result(hv=5.0):
    return SimpleNamespace(
        preferences=np.zeros((51, 2)),
        returns_per_repeat=np.zeros((3, 51, 2)),
        mean_hv=hv,
        source_hv=hv,
        mean_repeat_hv=hv - 0.5,
        mean_sparsity=1.0,
        source_sparsity=2.0,
        pareto_returns=np.zeros((3, 2)),
        source_pareto_point_count=3,
    )


def _append(path, threshold, actual, hv=5.0, run_id="seed15"):
    return append_convergence_result(
        path,
        seed=15,
        run_id=run_id,
        artifact_sha256="abc",
        target_transition_threshold=threshold,
        actual_env_transitions=actual,
        iteration=actual // 10,
        result=_result(hv),
        evaluation_seconds=1.25,
        wall_clock_seconds=8.5,
    )


def test_target_config_composes_with_unified_full_evaluation():
    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="experiment/pd_morl")
    assert config.hv_history.interval_transitions == 250_000
    assert "convergence_eval" not in config
    assert not hasattr(MOTD3Workflow, "_maybe_evaluate_convergence")


def test_reference_episode_trigger_is_retained():
    source = inspect.getsource(MOTD3Workflow._after_multi_steps)
    assert "interval_timesteps" not in source
    assert "full_evaluation_due" in source


def test_unified_full_evaluation_boundaries_resume_and_outputs(tmp_path):
    calls, snapshots = [], []
    workflow = PDMORLGPUWorkflow.__new__(PDMORLGPUWorkflow)
    workflow.config = OmegaConf.create({
        "seed": 15, "output_dir": str(tmp_path), "envs_per_preference_group": 16,
        "hv_history": {"enable": True, "write_csv": True, "interval_transitions": 250_000},
        # Even a stale override cannot re-enable the duplicate 51-preference evaluation.
        "convergence_eval": {"enable": True},
    })
    workflow.interpolator_artifact = SimpleNamespace(sha256="abc")

    def evaluate(agent_state, preferences, repeats):
        calls.append((agent_state, preferences.copy(), repeats))
        result = _result()
        result.preferences = preferences
        result.returns_per_repeat = np.zeros((repeats, len(preferences), 2))
        return result

    workflow.morl_evaluator = SimpleNamespace(evaluate=evaluate)
    workflow._write_evaluation_snapshot = lambda kind, result, iteration, **kw: snapshots.append(
        (kind, len(result.preferences), kw["sampled_timesteps"]))
    params, interpolator, rng, replay = object(), object(), object(), object()
    state = State(
        agent_state=PyTreeDict(params=params, extra_state=PyTreeDict(
            episode_count=jnp.full((10,), 16_000), eval_cnt_ep=jnp.uint32(1000),
            eval_cnt=jnp.uint32(1), interpolator=interpolator)),
        metrics=PyTreeDict(sampled_timesteps=jnp.uint32(249_999), iterations=jnp.uint32(9)),
        key=rng, replay_buffer_state=replay,
    )
    # Large episode counts cannot cause an early transition-based evaluation.
    workflow._prepare_evaluation_schedule(state)
    assert workflow._after_multi_steps(state)[0] is state
    assert not calls
    for sampled in (250_320, 500_080, 1_260_000):
        state = state.replace(metrics=state.metrics.replace(sampled_timesteps=jnp.uint32(sampled)))
        # Evaluation also works with zero completed episodes.
        state = state.replace(agent_state=state.agent_state.replace(
            extra_state=state.agent_state.extra_state.replace(episode_count=jnp.zeros((10,)))))
        state, _ = workflow._after_multi_steps(state)
        assert state.agent_state.params is params
        assert state.agent_state.extra_state.interpolator is interpolator
        assert state.key is rng and state.replay_buffer_state is replay
        workflow._after_multi_steps(state)  # The same boundary is not evaluated twice.
    assert len(calls) == 3
    assert all(p.shape == (201, 2) and n == 3 for _, p, n in calls)
    np.testing.assert_allclose(calls[0][1][[0, 100, 200]], [[0, 1], [.5, .5], [1, 0]])
    assert int(state.agent_state.extra_state.eval_cnt) == 4
    assert workflow._next_full_eval_transition == 1_500_000
    assert snapshots == [("training_full", 201, x) for x in (250_320, 500_080, 1_260_000)]
    with (tmp_path / "hv_history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["actual_env_transitions"]) for r in rows] == [250_320, 500_080, 1_260_000]
    assert all(r["preference_count"] == "201" and r["repeats"] == "3" for r in rows)
    assert rows[0]["source_hv"] == "5.0" and rows[0]["source_sparsity"] == "2.0"
    assert not (tmp_path / "hv_convergence.csv").exists()
    # A restored checkpoint initializes its next future threshold without backfilling.
    del workflow._next_full_eval_transition
    workflow._prepare_evaluation_schedule(state)
    assert workflow._next_full_eval_transition == 1_500_000
    workflow._after_multi_steps(state)
    assert len(calls) == 3
    for invalid in (0, -1):
        workflow.config.hv_history.interval_transitions = invalid
        with pytest.raises(ValueError, match="interval_transitions must be positive"):
            workflow._prepare_evaluation_schedule(state)
        with pytest.raises(ValueError, match="interval_transitions must be positive"):
            workflow._after_multi_steps(state)






def test_resume_uses_last_valid_threshold_and_archives_future_rows(tmp_path):
    path = tmp_path / "hv_convergence.csv"
    _append(path, 250_000, 250_320)
    _append(path, 500_000, 500_080)
    _append(path, 750_000, 750_160)

    next_due = prepare_convergence_history(
        path, run_id="seed15", sampled_timesteps=500_100, interval=250_000
    )

    assert next_due == 750_000
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["target_transition_threshold"]) for row in rows] == [
        250_000,
        500_000,
    ]
    archives = list(tmp_path.glob("hv_convergence.rollback-*.csv"))
    assert len(archives) == 1
    assert "750000" in archives[0].read_text(encoding="utf-8")


def test_empty_resume_does_not_backfill_old_thresholds(tmp_path):
    assert (
        prepare_convergence_history(
            tmp_path / "missing.csv",
            run_id="seed15",
            sampled_timesteps=760_000,
            interval=250_000,
        )
        == 1_000_000
    )




def test_convergence_csv_deduplicates_run_and_threshold(tmp_path):
    path = tmp_path / "hv_convergence.csv"
    assert _append(path, 250_000, 250_320)
    assert not _append(path, 250_000, 252_880)
    assert _append(path, 250_000, 252_880, run_id="seed651")
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert "evaluation_seconds" in rows[0]


def test_two_points_plot_and_no_points_summary(tmp_path):
    import pytest

    pytest.importorskip("matplotlib")
    path = tmp_path / "hv_convergence.csv"
    _append(path, 250_000, 250_320, hv=5.0)
    _append(path, 500_000, 500_080, hv=5.2)
    summary = plot_convergence_csv(path, tmp_path / "hv_convergence.png")
    assert summary["points"] == 2
    assert summary["status"] == "insufficient_data"
    assert (tmp_path / "hv_convergence.png").exists()
    assert (tmp_path / "hv_convergence_summary.json").exists()

    empty_summary = plot_convergence_csv(
        tmp_path / "missing.csv", tmp_path / "empty" / "curve.png"
    )
    assert empty_summary == {"status": "insufficient_data", "points": 0}
