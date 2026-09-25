from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp

from evorl.types import Action
from evorl.utils.jax_utils import rng_split

from ..env import Env, EnvState
from .wrapper import Wrapper


def sample_preference(
    key: chex.PRNGKey, batch_shape: Sequence[int] = ()
) -> chex.Array:
    """Temporarily sample uniformly from the two-objective simplex."""
    first = jax.random.uniform(key, shape=tuple(batch_shape))
    return jnp.stack((first, 1.0 - first), axis=-1)


class EpisodePreferenceWrapper(Wrapper):
    """Maintain one fixed preference per batched environment episode."""

    def reset(self, key: chex.PRNGKey) -> EnvState:
        env_key, preference_key = jax.random.split(key)
        state = self.env.reset(env_key)
        if state.done.ndim != 1:
            raise ValueError("EpisodePreferenceWrapper requires a batched environment")

        keys = jax.random.split(preference_key, state.done.shape[0])
        keys, sample_keys = rng_split(keys)
        preferences = jax.vmap(sample_preference)(sample_keys)

        return state.replace(
            info=state.info.replace(preference=preferences),
            _internal=state._internal.replace(preference_key=keys),
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        preferences = state.info.preference
        state = self.env.step(state, action)

        old_keys = state._internal.preference_key
        next_keys, sample_keys = rng_split(old_keys)
        candidates = jax.vmap(sample_preference)(sample_keys)
        done = state.done[..., None].astype(jnp.bool_)

        return state.replace(
            info=state.info.replace(
                preference=jnp.where(done, candidates, preferences)
            ),
            _internal=state._internal.replace(
                preference_key=jnp.where(done, next_keys, old_keys)
            ),
        )
