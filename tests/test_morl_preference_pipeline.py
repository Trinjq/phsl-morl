import jax
import jax.numpy as jnp

from evorl.envs import Box, Env, EnvState, create_wrapped_brax_env
from evorl.envs.wrappers import EpisodePreferenceWrapper, sample_preference
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


def test_preference_sampler():
    preference = jax.jit(sample_preference)(jax.random.PRNGKey(0))
    preferences = jax.jit(lambda key: sample_preference(key, (5,)))(
        jax.random.PRNGKey(1)
    )

    assert preference.shape == (2,)
    assert preferences.shape == (5, 2)
    _assert_simplex(preference)
    _assert_simplex(preferences)
    assert jnp.unique(preferences[:, 0]).shape[0] == 5


def test_episode_preference_lifecycle():
    dones = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    env = EpisodePreferenceWrapper(_FakeVmapEnv(dones))
    state = env.reset(jax.random.PRNGKey(2))
    initial = state.info.preference

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
    assert not jnp.array_equal(state.info.preference[1], initial[1])


def test_walker_preference_rollout_replay_and_time_limit():
    batch_size = 3
    rollout_length = 2
    env = create_wrapped_brax_env(
        "walker2d",
        episode_length=1,
        parallel=batch_size,
        vector_reward=True,
        episode_preference=True,
    )
    state = env.reset(jax.random.PRNGKey(5))
    initial = state.info.preference

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
    assert not jnp.array_equal(preferences[1], initial)

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
    test_preference_sampler()
    test_episode_preference_lifecycle()
    test_walker_preference_rollout_replay_and_time_limit()
    test_replay_preserves_preference_transition_pairs()
    _test_mo_walker2d_single_transition()
    _test_scalar_walker2d_reward_regression()
    print("all preference-pipeline checks passed")
