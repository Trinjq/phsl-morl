import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.envs import Box, Env, EnvState, create_wrapped_brax_env
from evorl.envs.wrappers import (
    EpisodePreferenceWrapper,
    make_logical_worker_ids,
    official_preference_grid,
    official_preference_subspaces,
    sample_official_preference,
    sample_preference,
)
from evorl.replay_buffers import ReplayBuffer
from evorl.rollout import env_step, rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import flatten_rollout_trajectory

from .test_morl_vector_reward import (
    _zero_actions,
    test_mo_walker2d_single_transition as _test_mo_walker2d_single_transition,
    test_scalar_walker2d_reward_regression as _test_scalar_walker2d_reward_regression,
)


class _FakeVmapEnv(Env):
    def __init__(self, dones):
        self._dones = dones
        self.num_envs = dones.shape[1]

    def reset(self, key):
        del key
        return EnvState(
            env_state=PyTreeDict(i=jnp.zeros((), dtype=jnp.int32)),
            obs=jnp.zeros((self.num_envs, 17)),
            reward=jnp.zeros((self.num_envs, 2)),
            done=jnp.zeros(self.num_envs),
        )

    def step(self, state, action):
        del action
        i = state.env_state.i
        return state.replace(
            env_state=state.env_state.replace(i=i + 1),
            reward=jnp.zeros((self.num_envs, 2)),
            done=self._dones[i],
        )

    @property
    def obs_space(self):
        return Box(low=-jnp.ones(17), high=jnp.ones(17))

    @property
    def action_space(self):
        return Box(low=-jnp.ones(6), high=jnp.ones(6))


def _assert_simplex(preferences):
    assert jnp.all(preferences >= 0)
    assert jnp.allclose(preferences.sum(axis=-1), 1.0)


def _numpy_official_grid():
    mesh = [np.arange(0, 1.001, 0.001) for _ in range(2)]
    grid = np.array(list(itertools.product(*mesh)))
    grid = grid[grid.sum(axis=1) == 1]
    return np.unique(grid, axis=0).astype(np.float32)


def test_temporary_uniform_preference_sampler():
    preference = jax.jit(sample_preference)(jax.random.PRNGKey(0))
    preferences = jax.jit(lambda key: sample_preference(key, (5,)))(
        jax.random.PRNGKey(1)
    )

    assert preference.shape == (2,)
    assert preferences.shape == (5, 2)
    _assert_simplex(preference)
    _assert_simplex(preferences)
    assert jnp.unique(preferences[:, 0]).shape[0] == 5


@pytest.mark.parametrize("process_count", [4, 8, 10, 16])
def test_official_preference_grid_and_subspaces(process_count):
    grid = official_preference_grid()
    numpy_grid = _numpy_official_grid()
    subspaces = official_preference_subspaces(process_count)
    numpy_subspaces = np.array_split(numpy_grid, process_count)

    assert grid.shape == (1001, 2)
    _assert_simplex(grid)
    assert jnp.array_equal(grid.sum(axis=-1), jnp.ones(1001))
    assert np.array_equal(np.asarray(grid), numpy_grid)
    assert len(subspaces) == process_count
    for subspace, numpy_subspace in zip(subspaces, numpy_subspaces):
        assert np.array_equal(np.asarray(subspace), numpy_subspace)


@pytest.mark.parametrize("process_count", [4, 8, 10, 16])
def test_official_worker_sampling(process_count):
    worker_ids = jnp.repeat(jnp.arange(process_count, dtype=jnp.int32), 128)
    keys = jax.random.split(jax.random.PRNGKey(9), worker_ids.shape[0])
    samples = jax.jit(
        jax.vmap(
            lambda key, worker_id: sample_official_preference(
                key, worker_id, process_count
            )
        )
    )(keys, worker_ids)
    indices = jnp.rint(samples[:, 0] * 1000).astype(jnp.int32)
    numpy_subspaces = np.array_split(np.arange(1001), process_count)
    starts = jnp.asarray([subspace[0] for subspace in numpy_subspaces])
    sizes = jnp.asarray([len(subspace) for subspace in numpy_subspaces])

    assert jnp.array_equal(
        make_logical_worker_ids(process_count, process_count),
        jnp.arange(process_count),
    )
    assert jnp.all(indices >= starts[worker_ids])
    assert jnp.all(indices < starts[worker_ids] + sizes[worker_ids])


@pytest.mark.parametrize(
    ("process_count", "batch_size"), [(4, 4), (4, 8), (8, 16)]
)
def test_logical_worker_ids_require_equal_replicas(process_count, batch_size):
    worker_ids = make_logical_worker_ids(batch_size, process_count)

    assert jnp.array_equal(
        worker_ids,
        jnp.arange(batch_size, dtype=jnp.int32) % process_count,
    )
    assert jnp.array_equal(
        jnp.bincount(worker_ids, length=process_count),
        jnp.full(process_count, batch_size // process_count),
    )


def test_logical_worker_ids_reject_unequal_replicas():
    with pytest.raises(ValueError, match=r"B=10, K=8"):
        create_wrapped_brax_env(
            "walker2d",
            parallel=10,
            episode_preference=True,
            process_count=8,
        )


def test_episode_preference_lifecycle():
    dones = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    worker_ids = jnp.arange(4, dtype=jnp.int32)
    env = EpisodePreferenceWrapper(_FakeVmapEnv(dones), worker_ids, process_count=4)
    state = env.reset(jax.random.PRNGKey(2))
    initial = state.info.preference
    initial_keys = state._internal.preference_key

    transition, state = jax.jit(
        lambda env_state: env_step(
            env.step,
            _zero_actions,
            env_state,
            None,
            jax.random.PRNGKey(3),
        )
    )(state)
    assert jnp.array_equal(transition.extras.policy_extras.preference, initial)
    assert jnp.array_equal(state.info.preference, initial)

    transition, state = jax.jit(
        lambda env_state: env_step(
            env.step,
            _zero_actions,
            env_state,
            None,
            jax.random.PRNGKey(4),
        )
    )(state)
    assert jnp.array_equal(transition.extras.policy_extras.preference, initial)
    assert jnp.array_equal(state.info.preference[0], initial[0])
    assert jnp.array_equal(state.info.preference[2], initial[2])
    assert jnp.array_equal(state.info.preference[3], initial[3])
    assert jnp.array_equal(state._internal.preference_key[0], initial_keys[0])
    assert jnp.array_equal(state._internal.preference_key[2], initial_keys[2])
    assert jnp.array_equal(state._internal.preference_key[3], initial_keys[3])
    assert not jnp.array_equal(state._internal.preference_key[1], initial_keys[1])


def test_walker_preference_rollout_replay_and_time_limit():
    batch_size = 4
    rollout_length = 2
    env = create_wrapped_brax_env(
        "walker2d",
        episode_length=1,
        parallel=batch_size,
        vector_reward=True,
        episode_preference=True,
        process_count=4,
    )
    state = env.reset(jax.random.PRNGKey(5))
    initial = state.info.preference
    initial_keys = state._internal.preference_key

    trajectory, state = jax.jit(
        lambda env_state: rollout(
            env.step,
            _zero_actions,
            env_state,
            None,
            jax.random.PRNGKey(6),
            rollout_length,
        )
    )(state)
    preferences = trajectory.extras.policy_extras.preference

    assert state.obs.shape == (batch_size, 17)
    assert trajectory.rewards.shape == (rollout_length, batch_size, 2)
    assert preferences.shape == (rollout_length, batch_size, 2)
    _assert_simplex(preferences)
    assert jnp.array_equal(preferences[0], initial)
    assert not jnp.array_equal(state._internal.preference_key, initial_keys)

    flat = flatten_rollout_trajectory(trajectory)
    replay = ReplayBuffer(capacity=16, sample_batch_size=4)
    replay_state = replay.init(flat.take(0))
    replay_state = jax.jit(replay.add)(replay_state, flat)
    sample = jax.jit(replay.sample)(replay_state, jax.random.PRNGKey(7))

    assert flat.rewards.shape == (rollout_length * batch_size, 2)
    assert flat.extras.policy_extras.preference.shape == (
        rollout_length * batch_size,
        2,
    )
    assert jnp.array_equal(
        replay_state.data.extras.policy_extras.preference[: flat.rewards.shape[0]],
        flat.extras.policy_extras.preference,
    )
    assert sample.rewards.shape == (4, 2)
    assert sample.extras.policy_extras.preference.shape == (4, 2)


def test_replay_preserves_preference_transition_pairs():
    ids = jnp.arange(4, dtype=jnp.float32)
    preferences = jnp.stack((ids / 4.0, 1.0 - ids / 4.0), axis=-1)
    transitions = SampleBatch(
        obs=ids[:, None],
        actions=ids[:, None],
        rewards=jnp.stack((ids, -ids), axis=-1),
        next_obs=(ids + 1)[:, None],
        dones=jnp.zeros(4),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(preference=preferences),
            env_extras=PyTreeDict(),
        ),
    )
    replay = ReplayBuffer(capacity=8, sample_batch_size=6)
    replay_state = replay.init(transitions.take(0))
    replay_state = jax.jit(replay.add)(replay_state, transitions)
    sample = jax.jit(replay.sample)(replay_state, jax.random.PRNGKey(8))
    sample_ids = sample.obs[:, 0].astype(jnp.int32)

    assert jnp.array_equal(
        sample.extras.policy_extras.preference, preferences[sample_ids]
    )
    assert jnp.array_equal(sample.rewards, transitions.rewards[sample_ids])


if __name__ == "__main__":
    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    test_temporary_uniform_preference_sampler()
    for process_count in (4, 8, 10, 16):
        test_official_preference_grid_and_subspaces(process_count)
        test_official_worker_sampling(process_count)
    for process_count, batch_size in ((4, 4), (4, 8), (8, 16)):
        test_logical_worker_ids_require_equal_replicas(process_count, batch_size)
    test_logical_worker_ids_reject_unequal_replicas()
    test_episode_preference_lifecycle()
    test_walker_preference_rollout_replay_and_time_limit()
    test_replay_preserves_preference_transition_pairs()
    _test_mo_walker2d_single_transition()
    _test_scalar_walker2d_reward_regression()
    print("all preference-pipeline checks passed")
