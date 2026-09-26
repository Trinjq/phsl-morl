import jax
import jax.numpy as jnp

from evorl.replay_buffers import ReplayBuffer
from evorl.replay_buffers.her import (
    add_her_transitions,
    sample_her_preferences,
)
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.jax_utils import tree_get


def _batch(ids):
    ids = jnp.asarray(ids, dtype=jnp.float32)
    batch_size = ids.shape[0]
    return SampleBatch(
        obs=jnp.stack((ids, ids + 0.1), axis=-1),
        actions=ids[:, None] + 1,
        rewards=jnp.stack((ids + 2, ids + 3), axis=-1),
        next_obs=jnp.stack((ids + 4, ids + 5), axis=-1),
        dones=(ids % 2).astype(jnp.float32),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(
                preference=jnp.tile(jnp.array([[0.25, 0.75]]), (batch_size, 1))
            ),
            env_extras=PyTreeDict(
                ori_obs=jnp.stack((ids + 4, ids + 5), axis=-1),
                termination=(ids % 2).astype(jnp.float32),
                truncation=jnp.zeros(batch_size),
            ),
        ),
    )


def _add(replay, state, batch, key, num_relabels=3, start=0, workers=1):
    return add_her_transitions(
        replay, state, batch, key, num_relabels, start, workers
    )


def _ordered(replay, state):
    size = int(state.buffer_size)
    if size < replay.capacity:
        indices = jnp.arange(size)
    else:
        indices = (state.current_index + jnp.arange(size)) % replay.capacity
    return tree_get(state.data, indices)


def test_official_her_sampler_jit_and_vmap():
    key = jax.random.PRNGKey(0)
    raw = jnp.abs(jax.random.normal(key, (3, 2)))
    expected = jnp.round(raw / jnp.sum(raw, axis=-1, keepdims=True), 3)
    sample = jax.jit(sample_her_preferences, static_argnums=(1, 2))(key, 3, 2)

    assert sample.shape == (3, 2)
    assert sample.dtype == jnp.float32
    assert jnp.array_equal(sample, expected)
    vmapped = jax.vmap(lambda k: sample_her_preferences(k, 3, 2))(
        jax.random.split(key, 4)
    )
    assert vmapped.shape == (4, 3, 2)


def test_original_relabels_and_transition_invariants():
    replay = ReplayBuffer(capacity=16, sample_batch_size=4)
    base = _batch([7])
    state = _add(replay, replay.init(base.take(0)), base, jax.random.PRNGKey(1))
    entries = _ordered(replay, state)

    assert int(state.buffer_size) == 4
    assert jnp.array_equal(
        entries.extras.policy_extras.preference[0],
        base.extras.policy_extras.preference[0],
    )
    assert not jnp.any(
        jnp.all(
            entries.extras.policy_extras.preference[1:]
            == base.extras.policy_extras.preference[0],
            axis=-1,
        )
    )
    for name in ("obs", "actions", "rewards", "next_obs", "dones"):
        values = getattr(entries, name)
        assert jnp.array_equal(values, jnp.repeat(values[:1], 4, axis=0))
    for name in ("ori_obs", "termination", "truncation"):
        values = getattr(entries.extras.env_extras, name)
        assert jnp.array_equal(values, jnp.repeat(values[:1], 4, axis=0))
    assert entries.rewards.shape == (4, 2)

    sampled = replay.sample(state, jax.random.PRNGKey(2))
    assert sampled.rewards.shape == (4, 2)
    assert sampled.extras.policy_extras.preference.shape == (4, 2)


def test_strict_warmup_boundary_and_process_count_parameterization():
    replay = ReplayBuffer(capacity=100, sample_batch_size=1)
    spec = _batch([0]).take(0)
    threshold = 10 * 4

    for initial_size, expected_size in (
        (threshold - 1, threshold),
        (threshold, threshold + 4),
        (threshold + 1, threshold + 5),
    ):
        state = replay.add(replay.init(spec), _batch(jnp.arange(initial_size)))
        state = _add(
            replay,
            state,
            _batch([99]),
            jax.random.PRNGKey(initial_size),
            start=10,
            workers=4,
        )
        assert int(state.buffer_size) == expected_size


def test_vectorized_batch_crosses_warmup_sequentially():
    replay = ReplayBuffer(capacity=32, sample_batch_size=1)
    spec = _batch([0]).take(0)
    state = replay.add(replay.init(spec), _batch(jnp.arange(8)))
    state = _add(
        replay,
        state,
        _batch([100, 101, 102, 103]),
        jax.random.PRNGKey(4),
        start=10,
        workers=1,
    )
    entries = _ordered(replay, state)

    assert int(state.buffer_size) == 18
    assert int(state.current_index) == 18
    added_ids = entries.obs[8:, 0]
    assert jnp.array_equal(
        added_ids,
        jnp.array([100, 101, 102, 102, 102, 102, 103, 103, 103, 103]),
    )
    original_indices = jnp.array([8, 9, 10, 14])
    assert jnp.array_equal(
        entries.extras.policy_extras.preference[original_indices],
        jnp.tile(jnp.array([[0.25, 0.75]]), (4, 1)),
    )


def test_num_relabel_preferences_parameterization_and_jitted_add():
    for num_relabels in (0, 1, 3):
        replay = ReplayBuffer(capacity=16, sample_batch_size=1)
        batch = _batch([1])
        add = jax.jit(
            lambda state, xs, key: _add(
                replay, state, xs, key, num_relabels=num_relabels
            )
        )
        state = add(replay.init(batch.take(0)), batch, jax.random.PRNGKey(3))
        assert int(state.buffer_size) == 1 + num_relabels


def test_full_ring_buffer_matches_source_group_order_and_eviction():
    replay = ReplayBuffer(capacity=8, sample_batch_size=1)
    spec = _batch([0]).take(0)
    state = replay.init(spec)
    for transition_id in range(3):
        state = _add(
            replay,
            state,
            _batch([transition_id]),
            jax.random.PRNGKey(10 + transition_id),
        )

    entries = _ordered(replay, state)
    assert int(state.buffer_size) == 8
    assert int(state.current_index) == 4
    assert jnp.array_equal(entries.obs[:, 0], jnp.repeat(jnp.array([1.0, 2.0]), 4))
    assert jnp.array_equal(
        entries.extras.policy_extras.preference[::4],
        jnp.tile(jnp.array([[0.25, 0.75]]), (2, 1)),
    )
