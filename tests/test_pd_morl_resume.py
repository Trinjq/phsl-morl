import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from omegaconf import OmegaConf
from train import _resolve_resume_checkpoint, _restore_checkpoint


def test_resume_latest_selects_highest_numeric_checkpoint(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "100").mkdir(parents=True)
    (checkpoints / "200").mkdir()
    (checkpoints / "not-a-step").mkdir()
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "resume_from_checkpoint": None,
            "resume_latest": True,
        }
    )
    assert _resolve_resume_checkpoint(config) == checkpoints / "200"


def test_resume_rejects_checkpoint_without_replay_buffer(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "100"
    checkpoint.mkdir(parents=True)
    config = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "resume_from_checkpoint": str(checkpoint),
            "resume_latest": False,
            "save_replay_buffer": False,
        }
    )
    with pytest.raises(ValueError, match="save_replay_buffer=true"):
        _restore_checkpoint(config, object(), object(), {})


def test_restore_uses_rollout_units_and_trims_future_history(tmp_path, monkeypatch):
    source = tmp_path / "source"
    checkpoint = source / "checkpoints" / "8"
    checkpoint.mkdir(parents=True)
    with (source / "hv_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["actual_env_transitions", "source_hv"])
        writer.writeheader()
        writer.writerows(
            [
                {"actual_env_transitions": 100, "source_hv": 1},
                {"actual_env_transitions": 300, "source_hv": 2},
            ]
        )
    (source / "training_diagnostics.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"base_inserts": 100, "key_replacement_count": 1},
                {"base_inserts": 300, "key_replacement_count": 9},
            )
        ),
        encoding="utf-8",
    )

    restored = SimpleNamespace(
        metrics=SimpleNamespace(sampled_timesteps=200, iterations=8),
        agent_state=SimpleNamespace(
            extra_state=SimpleNamespace(total_it=80, eval_cnt_ep=3)
        ),
    )
    monkeypatch.setattr("evorl.utils.orbax_utils.load", lambda *_args: restored)
    workflow = SimpleNamespace(runtime_counters={})
    target = tmp_path / "resumed"
    config = OmegaConf.create(
        {
            "output_dir": str(target),
            "resume_from_checkpoint": str(checkpoint),
            "resume_latest": False,
            "save_replay_buffer": True,
            "fold_iters": 4,
            "actor_update_interval": 10,
        }
    )

    assert _restore_checkpoint(config, workflow, object(), {}) is restored
    assert workflow.runtime_counters["host_chunk_count"] == 2
    assert workflow.runtime_counters["inner_rollout_count"] == 8
    assert workflow.runtime_counters["key_replacement_count"] == 1
    with (target / "hv_history.csv").open(newline="", encoding="utf-8") as f:
        assert [int(row["actual_env_transitions"]) for row in csv.DictReader(f)] == [
            100
        ]


@pytest.mark.parametrize("layout", ["direct", "managed_path", "managed_api"])
def test_checkpoint_roundtrip_preserves_empty_state_leaves(tmp_path, layout):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from evorl.types import State
    from evorl.utils.orbax_utils import load, save, setup_checkpoint_manager

    state = State(params=jnp.arange(3.0), empty=jnp.zeros((0, 2)))
    target = state.replace(params=jnp.zeros(3))
    if layout == "direct":
        path = tmp_path / "direct"
        save(path, state)
        restored = load(path, target)
    else:
        config = OmegaConf.create({
            "output_dir": str(tmp_path),
            "checkpoint": {"enable": True, "save_interval_steps": 1, "max_to_keep": 2},
        })
        with setup_checkpoint_manager(config) as manager:
            manager.save(1, state, force=True)
            manager.wait_until_finished()
            restored = (
                manager.restore(1, items=target) if layout == "managed_api"
                else load(tmp_path / "checkpoints" / "1", target)
            )
    assert jax.tree.structure(restored) == jax.tree.structure(state)
    np.testing.assert_array_equal(restored.params, state.params)
    assert restored.empty.shape == (0, 2)
