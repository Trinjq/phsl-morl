"""Source-faithful PD-MORL preference relabeling for JAX replay writes."""

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.sample_batch import SampleBatch

from .replay_buffer import ReplayBuffer, ReplayBufferState


def sample_her_preferences(
    key: chex.PRNGKey, num_preferences: int, objective_size: int
) -> chex.Array:
    """Sample abs-normal, L1-normalized preferences rounded to 3 decimals."""
    values = jnp.abs(jax.random.normal(key, (num_preferences, objective_size)))
    norm = jnp.sum(values, axis=-1, keepdims=True)
    norm = jnp.maximum(norm, jnp.finfo(values.dtype).tiny)
    return jnp.round(values / norm, decimals=3)


def expand_her_transitions(
    transitions: SampleBatch,
    key: chex.PRNGKey,
    num_relabel_preferences: int,
    active: chex.Array,
) -> tuple[SampleBatch, chex.Array]:
    """Expand base transitions in original-then-relabeled insertion order."""
    batch_size, objective_size = transitions.rewards.shape[:2]
    group_size = 1 + num_relabel_preferences
    keys = jax.random.split(key, batch_size)
    relabeled = jax.vmap(
        lambda sample_key: sample_her_preferences(
            sample_key, num_relabel_preferences, objective_size
        )
    )(keys)

    expanded = jtu.tree_map(
        lambda x: jnp.repeat(x[:, None, ...], group_size, axis=1), transitions
    )
    preferences = jnp.concatenate(
        (
            transitions.extras.policy_extras.preference[:, None, :],
            relabeled,
        ),
        axis=1,
    )
    expanded = expanded.replace(
        extras=expanded.extras.replace(
            policy_extras=expanded.extras.policy_extras.replace(
                preference=preferences
            )
        )
    )
    expanded = jtu.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])), expanded
    )
    mask = jnp.concatenate(
        (
            jnp.ones((batch_size, 1), dtype=jnp.bool_),
            jnp.repeat(active[:, None], num_relabel_preferences, axis=1),
        ),
        axis=1,
    )
    return expanded, mask.reshape(-1)


def add_her_transitions(
    replay_buffer: ReplayBuffer,
    buffer_state: ReplayBufferState,
    transitions: SampleBatch,
    key: chex.PRNGKey,
    num_relabel_preferences: int,
    start_timesteps: int,
    process_count: int,
) -> ReplayBufferState:
    """Physically add original and active HER entries to one replay pool."""
    batch_size = transitions.rewards.shape[0]
    threshold = start_timesteps * process_count

    def activation_step(size, unused):
        del unused
        active = (size >= replay_buffer.capacity) | (size + 1 > threshold)
        added = 1 + active.astype(jnp.int32) * num_relabel_preferences
        return jnp.minimum(size + added, replay_buffer.capacity), active

    _, active = jax.lax.scan(
        activation_step, buffer_state.buffer_size, None, length=batch_size
    )
    expanded, mask = expand_her_transitions(
        transitions, key, num_relabel_preferences, active
    )
    return replay_buffer.add(buffer_state, expanded, mask)
