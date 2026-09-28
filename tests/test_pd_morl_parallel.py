from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from evorl.algorithms.mo_td3 import (
    make_mo_td3_agent,
    parallel_actor_update_mask,
)
from evorl.algorithms.offpolicy_utils import clean_trajectory
from evorl.envs import Box, EnvState
from evorl.metrics import WorkflowMetric
from evorl.replay_buffers import ReplayBuffer
from evorl.replay_buffers.her import add_her_transitions
from evorl.rollout import rollout
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.rl_toolkits import flatten_rollout_trajectory
from omegaconf import OmegaConf


def _env_state(process_count):
    worker_ids = jnp.arange(process_count, dtype=jnp.float32)
    preferences = jnp.stack(
        (worker_ids / process_count, 1 - worker_ids / process_count), axis=-1
    )
    return EnvState(
        env_state=PyTreeDict(),
        obs=worker_ids[:, None],
        reward=jnp.zeros((process_count, 2)),
        done=jnp.zeros(process_count),
        info=PyTreeDict(
            preference=preferences,
            ori_obs=worker_ids[:, None],
            termination=jnp.zeros(process_count),
            truncation=jnp.zeros(process_count),
            actions=jnp.zeros((process_count, 1)),
        ),
    )


def _env_step(state, actions):
    worker_ids = state.obs[:, 0]
    next_obs = state.obs + 0.5
    return state.replace(
        obs=next_obs,
        reward=jnp.stack((worker_ids, -worker_ids), axis=-1),
        info=state.info.replace(
            ori_obs=next_obs,
            termination=jnp.zeros_like(worker_ids),
            truncation=jnp.zeros_like(worker_ids),
            actions=actions,
        ),
    )


def _shared_policy(agent_state, sample_batch, key):
    del key
    actions = sample_batch.obs * agent_state.actor_scale
    return actions, PyTreeDict(
        preference=sample_batch.extras.policy_extras.preference
    )


def _collect(state, process_count):
    trajectory, next_state = rollout(
        _env_step,
        _shared_policy,
        state,
        PyTreeDict(actor_scale=jnp.array(2.0)),
        jax.random.PRNGKey(process_count),
        1,
        env_extra_fields=("ori_obs", "termination", "truncation", "actions"),
    )
    return flatten_rollout_trajectory(clean_trajectory(trajectory)), next_state


@pytest.mark.parametrize("process_count", [4, 8, 10])
def test_parallel_collection_shared_policy_global_replay_and_her(process_count):
    state = _env_state(process_count)
    transitions, _ = jax.jit(lambda x: _collect(x, process_count))(state)
    replay = ReplayBuffer(capacity=64, sample_batch_size=process_count * 4)
    replay_state = replay.init(transitions.take(0))
    replay_state = jax.jit(
        lambda rb_state, batch: add_her_transitions(
            replay,
            rb_state,
            batch,
            jax.random.PRNGKey(100 + process_count),
            3,
            0,
            process_count,
        )
    )(replay_state, transitions)

    assert transitions.obs.shape == (process_count, 1)
    assert jnp.array_equal(
        transitions.obs[:, 0], jnp.arange(process_count, dtype=jnp.float32)
    )
    assert jnp.array_equal(transitions.actions, transitions.obs * 2)
    assert int(replay_state.buffer_size) == process_count * 4
    stored_ids = replay_state.data.obs[: process_count * 4, 0]
    assert jnp.array_equal(
        stored_ids, jnp.repeat(jnp.arange(process_count), 4)
    )
    assert jnp.array_equal(
        replay_state.data.actions[: process_count * 4],
        jnp.repeat(transitions.actions, 4, axis=0),
    )
    assert jnp.array_equal(
        replay_state.data.rewards[: process_count * 4],
        jnp.repeat(transitions.rewards, 4, axis=0),
    )
    assert jnp.array_equal(
        replay_state.data.extras.policy_extras.preference[
            : process_count * 4 : 4
        ],
        transitions.extras.policy_extras.preference,
    )
    sample = replay.sample(replay_state, jax.random.PRNGKey(200 + process_count))
    assert jnp.unique(sample.obs[:, 0]).size > 1


def test_parallel_collection_preserves_strict_warmup_crossing():
    process_count = 4
    transitions, _ = _collect(_env_state(process_count), process_count)
    replay = ReplayBuffer(capacity=32, sample_batch_size=1)
    replay_state = replay.init(transitions.take(0))
    replay_state = replay.add(replay_state, transitions.concatenate(transitions))
    replay_state = add_her_transitions(
        replay,
        replay_state,
        transitions,
        jax.random.PRNGKey(5),
        3,
        10,
        1,
    )

    assert int(replay_state.buffer_size) == 18
    added_ids = replay_state.data.obs[8:18, 0]
    assert jnp.array_equal(added_ids, jnp.array([0, 1, 2, 2, 2, 2, 3, 3, 3, 3]))


def test_official_learner_start_boundary():
    config = OmegaConf.load(
        Path(__file__).parents[1] / "configs" / "agent" / "mo-td3.yaml"
    )
    threshold = 2 * config.batch_size * config.num_relabel_preferences
    replay = ReplayBuffer(
        capacity=threshold,
        min_sample_timesteps=config.learner_start_replay_entries,
        sample_batch_size=config.batch_size,
    )
    transitions, _ = _collect(_env_state(4), 4)
    replay_state = replay.init(transitions.take(0))

    assert threshold == 1536
    assert config.learning_start_timesteps == 1530
    assert config.start_timesteps == 10000
    assert config.her_start_timesteps == 10000
    assert not replay.can_sample(
        replay_state.replace(buffer_size=jnp.array(threshold - 1))
    )
    assert replay.can_sample(
        replay_state.replace(buffer_size=jnp.array(threshold))
    )


def test_per_worker_random_action_warmup_boundary_jit():
    process_count = 10
    action_space = Box(
        low=jnp.array([-2.0, -1.0]), high=jnp.array([2.0, 3.0])
    )
    obs_space = Box(low=-jnp.ones(3), high=jnp.ones(3))
    agent = make_mo_td3_agent(
        action_space,
        actor_hidden_layer_sizes=(8,),
        critic_hidden_layer_sizes=(8,),
        exploration_epsilon=0.0,
        process_count=process_count,
        start_timesteps=10000,
    )
    agent_state = agent.init(
        obs_space, action_space, jax.random.PRNGKey(10)
    )
    worker_steps = jnp.array(
        [9999, 10000, 9998, 10001, 9999, 10000, 0, 12000, 9999, 10000],
        dtype=jnp.uint32,
    )
    agent_state = agent_state.replace(
        extra_state=agent_state.extra_state.replace(worker_steps=worker_steps)
    )
    preference = jnp.tile(jnp.array([[0.25, 0.75]]), (process_count, 1))
    batch = SampleBatch(
        obs=jnp.zeros((process_count, 3)),
        extras=PyTreeDict(
            policy_extras=PyTreeDict(preference=preference)
        ),
    )
    key = jax.random.PRNGKey(11)
    actions, extras = jax.jit(agent.compute_actions)(agent_state, batch, key)
    evaluated, _ = agent.evaluate_actions(agent_state, batch, key)
    assert jnp.isfinite(evaluated).all()
    assert jnp.all(evaluated >= action_space.low)
    assert jnp.all(evaluated <= action_space.high)
    expected_policy = jnp.clip(evaluated, action_space.low, action_space.high)
    _, random_key = jax.random.split(key)
    expected_random = jax.random.uniform(
        random_key,
        actions.shape,
        minval=action_space.low,
        maxval=action_space.high,
    )
    random_mask = worker_steps < 10000

    assert jnp.array_equal(actions[random_mask], expected_random[random_mask])
    assert jnp.array_equal(actions[~random_mask], expected_policy[~random_mask])
    assert jnp.all(actions >= action_space.low)
    assert jnp.all(actions <= action_space.high)
    assert jnp.array_equal(extras.preference, preference)


@pytest.mark.parametrize("process_count", [4, 8, 10])
def test_source_update_schedule_depends_on_process_count(process_count):
    masks = jax.jit(
        jax.vmap(
            lambda round_index: parallel_actor_update_mask(
                round_index, process_count, 10
            )
        )
    )(jnp.arange(10))

    assert masks.shape == (10, process_count)
    assert int(masks.sum()) == process_count
    flat_masks = masks.reshape(-1)
    assert jnp.all(flat_masks[9::10])
    assert not jnp.any(
        flat_masks[jnp.arange(flat_masks.size) % 10 != 9]
    )


def test_delayed_update_starts_at_first_real_learner_round():
    metrics_after_153_collection_rounds = WorkflowMetric(
        sampled_timesteps=jnp.uint32(1530)
    )
    first_update_ids = (
        metrics_after_153_collection_rounds.iterations * 10
        + jnp.arange(1, 11)
    )
    mask = parallel_actor_update_mask(
        metrics_after_153_collection_rounds.iterations, 10, 10
    )

    assert metrics_after_153_collection_rounds.iterations == 0
    assert jnp.array_equal(first_update_ids, jnp.arange(1, 11))
    assert jnp.array_equal(
        mask, jnp.array([False] * 9 + [True])
    )
