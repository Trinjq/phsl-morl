import csv
import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from evorl.algorithms.mo_td3 import MOTD3Workflow
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


def test_target_config_composes_with_independent_protocols():
    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name="experiment/pd_morl_brax_gpu_native_v2")
    assert "interval_timesteps" not in config.hv_history
    assert config.convergence_eval.interval_transitions == 250_000
    assert config.convergence_eval.preference_step == 0.02
    assert config.convergence_eval.repeats == 3


def test_formal_trigger_has_no_convergence_override():
    source = inspect.getsource(MOTD3Workflow._after_multi_steps)
    assert "interval_timesteps" not in source
    assert "full_evaluation_due" in source


def test_disabled_convergence_does_not_evaluate(tmp_path):
    workflow = MOTD3Workflow.__new__(MOTD3Workflow)
    workflow.config = OmegaConf.create(
        {"output_dir": str(tmp_path), "convergence_eval": {"enable": False}}
    )
    workflow.morl_evaluator = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError())
    )
    state = SimpleNamespace()
    assert workflow._maybe_evaluate_convergence(state) == {}


def test_convergence_boundary_crossing_is_single_and_read_only(tmp_path):
    calls = []

    class Evaluator:
        def evaluate(self, agent_state, preferences, repeats):
            calls.append((agent_state, preferences.copy(), repeats))
            return _result()

    workflow = MOTD3Workflow.__new__(MOTD3Workflow)
    workflow.config = OmegaConf.create(
        {
            "seed": 15,
            "output_dir": str(tmp_path),
            "convergence_eval": {
                "enable": True,
                "interval_transitions": 250_000,
                "preference_step": 0.02,
                "repeats": 3,
                "write_csv": True,
                "console_output": False,
            },
        }
    )
    workflow.morl_evaluator = Evaluator()
    workflow.interpolator_artifact = SimpleNamespace(sha256="abc")
    workflow._next_convergence_threshold = 250_000
    agent_state = object()
    state = SimpleNamespace(
        agent_state=agent_state,
        metrics=SimpleNamespace(
            sampled_timesteps=jnp.uint32(800_000), iterations=jnp.uint32(9)
        ),
    )

    assert workflow._maybe_evaluate_convergence(state) == {}
    assert state.agent_state is agent_state
    assert len(calls) == 1
    assert calls[0][1].shape == (51, 2)
    assert calls[0][2] == 3
    assert workflow._next_convergence_threshold == 1_000_000
    assert workflow._maybe_evaluate_convergence(state) == {}
    assert len(calls) == 1
    with (tmp_path / "hv_convergence.csv").open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["target_transition_threshold"] == "250000"
    assert row["actual_env_transitions"] == "800000"


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


def test_resume_keeps_existing_run_id(tmp_path):
    _append(tmp_path / "hv_convergence.csv", 250_000, 250_320, run_id="original")
    workflow = MOTD3Workflow.__new__(MOTD3Workflow)
    workflow.config = OmegaConf.create({"output_dir": str(tmp_path)})
    assert workflow._convergence_run_id() == "original"


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
