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
            "convergence_eval": {"enable": False},
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
