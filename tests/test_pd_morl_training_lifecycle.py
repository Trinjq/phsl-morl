from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from evorl.algorithms.mo_td3 import PDMORLGPUWorkflow, training_chunk_plan
from omegaconf import OmegaConf


class _CheckpointManager:
    def __init__(self, events):
        self.events = events

    def latest_step(self):
        return None

    def save(self, step, state, force=False):
        self.events.append(("save", step, force))
        return True

    def wait_until_finished(self):
        self.events.append(("wait",))


def _workflow(events, *, final_evaluation):
    workflow = PDMORLGPUWorkflow.__new__(PDMORLGPUWorkflow)
    workflow.config = OmegaConf.create(
        {
            "run_final_evaluation": final_evaluation,
            "output_dir": ".",
            "save_replay_buffer": True,
            "checkpoint": {"enable": True},
            "hv_history": {"enable": False},
        }
    )
    workflow.checkpoint_manager = _CheckpointManager(events)
    return workflow


def _state():
    return SimpleNamespace(
        metrics=SimpleNamespace(
            iterations=jnp.uint32(12), sampled_timesteps=jnp.uint32(1_000)
        ),
        agent_state=object(),
    )


def test_chunk_plan_uses_rollout_iteration_units():
    assert training_chunk_plan(10_000_960, 4_160, 0, 2_560, 4) == (3_905, 15_620)
    assert training_chunk_plan(10_003_520, 4_160, 0, 5_120, 4) == (1_953, 7_812)
    assert training_chunk_plan(1_000, 1_000, 20, 2_560, 4) == (0, 20)


def test_final_checkpoint_is_saved_when_final_evaluation_is_disabled():
    events = []
    state = _state()
    assert _workflow(events, final_evaluation=False)._after_learning(state) is state
    assert events == [("save", 12, True), ("wait",)]


def test_final_checkpoint_precedes_evaluation_failure():
    events = []
    workflow = _workflow(events, final_evaluation=True)

    def fail(*_args, **_kwargs):
        events.append(("evaluate",))
        raise RuntimeError("evaluation failed")

    workflow.morl_evaluator = SimpleNamespace(evaluate=fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        workflow._after_learning(_state())
    assert events == [("save", 12, True), ("wait",), ("evaluate",)]


@pytest.mark.parametrize("removed_option", [None, False, True])
def test_resume_rejects_enabled_unsupported_network_options(
    tmp_path, monkeypatch, removed_option
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from train import _restore_checkpoint

    source = tmp_path / "source"
    checkpoint = source / "checkpoints/8"
    checkpoint.mkdir(parents=True)
    network = {"critic_hidden_layer_sizes": [400, 400]}
    saved_network = dict(network)
    if removed_option is not None:
        saved_network["removed_network_option"] = removed_option
    OmegaConf.save(
        OmegaConf.create({"agent_network": saved_network}),
        source / "resolved_config.yaml",
    )
    config = OmegaConf.create({
        "resume_from_checkpoint": str(checkpoint), "save_replay_buffer": True,
        "output_dir": str(tmp_path / "target"), "agent_network": network,
    })
    restored = SimpleNamespace(metrics=SimpleNamespace(sampled_timesteps=100))
    calls = []

    def load(*_):
        calls.append(True)
        return restored

    monkeypatch.setattr("evorl.utils.orbax_utils.load", load)
    if removed_option:
        with pytest.raises(ValueError, match="unsupported agent_network settings"):
            _restore_checkpoint(config, object(), object(), {})
        assert not calls
    else:
        assert _restore_checkpoint(config, object(), object(), {}) is restored
        assert calls
