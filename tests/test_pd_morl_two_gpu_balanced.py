from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from evorl.distributed import (
    PDMORLDeviceLayout,
    logical_worker_keys,
    make_distributed_gradient_step,
)
from evorl.replay_buffers import ReplayBuffer
from evorl.utils.orbax_utils import load, save

from tests.pd_morl_multi_gpu_smoke import run_real_walker_distributed_smoke

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.multi_gpu,
    pytest.mark.slow,
    pytest.mark.integration,
]


def _gpu_devices(count):
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if len(devices) != count:
        pytest.skip(f"requires exactly {count} visible GPUs")
    return devices


def _loss(params, batch, mask):
    per_row = jnp.square(batch @ params - 1.0)
    contribution = jnp.where(mask, per_row, 0).sum() / 256
    return contribution, {"squared_error": contribution}


def test_two_gpu_balanced_learner_global_replay_and_checkpoint(tmp_path: Path):
    devices = _gpu_devices(2)
    layout = PDMORLDeviceLayout(2)
    assert layout.logical_worker_ids.reshape(2, 5).tolist() == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]
    assert layout.physical_rollout_slots == layout.logical_workers == 10
    assert layout.physical_learner_batch == layout.global_batch_size == 256
    assert layout.learner_rows_per_device == 128
    assert layout.valid_learner_mask.all()

    batch = jnp.arange(256 * 3, dtype=jnp.float32).reshape(256, 3) / 100
    params = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(100), optax.adam(3e-4))
    opt_state = optimizer.init(params)
    (single_loss, _), single_grads = jax.value_and_grad(_loss, has_aux=True)(
        params, batch, layout.valid_learner_mask
    )
    updates, single_opt_state = optimizer.update(single_grads, opt_state, params)
    single_params = optax.apply_updates(params, updates)

    step = make_distributed_gradient_step(_loss, optimizer, devices)
    loss, _, grads, updated, updated_opt_state = step(
        opt_state, params, batch, layout.valid_learner_mask
    )
    np.testing.assert_allclose(loss, single_loss, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(grads, single_grads, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(updated, single_params, rtol=2e-6, atol=2e-6)
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-6),
        updated_opt_state,
        single_opt_state,
    )
    for shard in updated.addressable_shards:
        np.testing.assert_array_equal(shard.data, updated)

    replay = ReplayBuffer(capacity=32, sample_batch_size=10)
    replay_state = replay.init(jnp.asarray(-1, dtype=jnp.int32))
    replay_state = replay.add(replay_state, jnp.arange(10, dtype=jnp.int32))
    assert int(replay_state.buffer_size) == 10
    np.testing.assert_array_equal(replay_state.data[:10], jnp.arange(10))

    keys = logical_worker_keys(jax.random.PRNGKey(7), jnp.arange(10))
    remapped = logical_worker_keys(jax.random.PRNGKey(7), jnp.arange(10)[::-1])
    np.testing.assert_array_equal(keys, remapped[::-1])

    checkpoint = tmp_path / "two_gpu_checkpoint"
    save(checkpoint, {"params": updated, "opt_state": updated_opt_state})
    restored = load(checkpoint, {"params": updated, "opt_state": updated_opt_state})
    np.testing.assert_array_equal(restored["params"], updated)


def test_two_gpu_real_walker_production_shape_smoke(tmp_path: Path):
    diagnostics = run_real_walker_distributed_smoke(_gpu_devices(2), tmp_path)
    assert diagnostics["rollout_slots"] == diagnostics["valid_rollout"] == 10
    assert diagnostics["learner_rows"] == diagnostics["valid_learner"] == 256
    assert diagnostics["replay_size"] == 40
