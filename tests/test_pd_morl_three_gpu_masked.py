import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from evorl.distributed import (
    PDMORLDeviceLayout,
    logical_worker_keys,
    make_distributed_gradient_step,
    masked_mean,
    pad_batch,
)
from evorl.replay_buffers import ReplayBuffer

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


def test_three_gpu_masked_learner_padding_has_zero_effect():
    devices = _gpu_devices(3)
    layout = PDMORLDeviceLayout(3)
    assert layout.logical_worker_ids.reshape(3, 4).tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, -1, -1],
    ]
    assert layout.physical_rollout_slots == 12
    assert int(layout.valid_rollout_mask.sum()) == 10
    assert layout.physical_learner_batch == 258
    assert layout.learner_rows_per_device == 86
    assert layout.valid_learner_mask.reshape(3, 86).sum(axis=1).tolist() == [86, 86, 84]

    valid_batch = jnp.arange(256 * 3, dtype=jnp.float32).reshape(256, 3) / 100
    padded_batch = pad_batch(valid_batch, layout.physical_learner_batch)
    changed_padding = padded_batch.at[-2:].set(jnp.array([[1e6] * 3, [-1e6] * 3]))
    params = jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(100), optax.adam(3e-4))
    opt_state = optimizer.init(params)

    single_mask = jnp.ones(256, dtype=jnp.bool_)
    (single_loss, _), single_grads = jax.value_and_grad(_loss, has_aux=True)(
        params, valid_batch, single_mask
    )
    updates, single_opt_state = optimizer.update(single_grads, opt_state, params)
    single_params = optax.apply_updates(params, updates)

    step = make_distributed_gradient_step(_loss, optimizer, devices)
    actual = step(opt_state, params, padded_batch, layout.valid_learner_mask)
    changed = step(opt_state, params, changed_padding, layout.valid_learner_mask)
    loss, _, grads, updated, updated_opt_state = actual
    np.testing.assert_allclose(loss, single_loss, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(grads, single_grads, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(updated, single_params, rtol=2e-6, atol=2e-6)
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=2e-6, atol=2e-6),
        updated_opt_state,
        single_opt_state,
    )
    jax.tree.map(
        lambda x, y: np.testing.assert_allclose(x, y, rtol=0, atol=0),
        actual,
        changed,
    )
    assert (
        masked_mean(jnp.arange(258.0), layout.valid_learner_mask)
        == jnp.arange(256.0).mean()
    )

    replay = ReplayBuffer(capacity=32, sample_batch_size=10)
    replay_state = replay.init(jnp.asarray(-1, dtype=jnp.int32))
    replay_state = replay.add(
        replay_state, layout.logical_worker_ids, mask=layout.valid_rollout_mask
    )
    assert int(replay_state.buffer_size) == 10
    np.testing.assert_array_equal(replay_state.data[:10], jnp.arange(10))

    keys = logical_worker_keys(jax.random.PRNGKey(11), jnp.arange(10))
    placement = jnp.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
    moved = logical_worker_keys(jax.random.PRNGKey(11), placement)
    np.testing.assert_array_equal(moved, keys[placement])


def test_three_gpu_real_walker_production_shape_smoke(tmp_path):
    diagnostics = run_real_walker_distributed_smoke(_gpu_devices(3), tmp_path)
    assert diagnostics["rollout_slots"] == 12
    assert diagnostics["valid_rollout"] == 10
    assert diagnostics["learner_rows"] == 258
    assert diagnostics["valid_learner"] == 256
    assert diagnostics["replay_size"] == 40
