from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp

from evorl.types import Action
from evorl.utils.jax_utils import rng_split

from ..env import Env, EnvState
from .wrapper import Wrapper

PREFERENCE_GRID_SIZE = 1001
_PREFERENCE_VALUES = tuple(i / 1000 for i in range(PREFERENCE_GRID_SIZE))


def _preference_subspace_bounds(
    process_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(process_count, int) or isinstance(process_count, bool):
        raise ValueError("process_count must be an integer")
    if not 1 <= process_count <= PREFERENCE_GRID_SIZE:
        raise ValueError(f"process_count must be between 1 and {PREFERENCE_GRID_SIZE}")

    size, remainder = divmod(PREFERENCE_GRID_SIZE, process_count)
    sizes = tuple(size + (worker_id < remainder) for worker_id in range(process_count))
    starts = tuple(
        worker_id * size + min(worker_id, remainder)
        for worker_id in range(process_count)
    )
    return starts, sizes


def make_logical_worker_ids(env_batch_size: int, process_count: int = 10) -> chex.Array:
    """Map equally replicated environment lanes onto logical workers."""
    _preference_subspace_bounds(process_count)
    if env_batch_size % process_count != 0:
        raise ValueError(
            "env batch size must be divisible by process_count: "
            f"got B={env_batch_size}, K={process_count}"
        )
    return jnp.arange(env_batch_size, dtype=jnp.int32) % process_count


def sample_preference(key: chex.PRNGKey, batch_shape: Sequence[int] = ()) -> chex.Array:
    """Sample continuously from the simplex for tests only."""
    first = jax.random.uniform(key, shape=tuple(batch_shape))
    return jnp.stack((first, 1.0 - first), axis=-1)


def official_preference_grid() -> chex.Array:
    """Return the official ordered two-objective 0.001 preference grid."""
    values = jnp.asarray(_PREFERENCE_VALUES, dtype=jnp.float32)
    return jnp.stack((values, values[::-1]), axis=-1)


def official_preference_subspaces(
    process_count: int = 10,
) -> tuple[chex.Array, ...]:
    """Split the official grid like np.array_split(grid, process_count)."""
    grid = official_preference_grid()
    starts, sizes = _preference_subspace_bounds(process_count)
    return tuple(grid[start : start + size] for start, size in zip(starts, sizes))


def sample_official_preference(
    key: chex.PRNGKey,
    logical_worker_id: chex.Array,
    process_count: int = 10,
) -> chex.Array:
    """Sample from one logical worker's official preference subspace."""
    starts, sizes = _preference_subspace_bounds(process_count)
    starts = jnp.asarray(starts, dtype=jnp.int32)
    sizes = jnp.asarray(sizes, dtype=jnp.int32)
    start = starts[logical_worker_id]
    index = start + jax.random.randint(
        key, (), minval=0, maxval=sizes[logical_worker_id]
    )
    return official_preference_grid()[index]


class EpisodePreferenceWrapper(Wrapper):
    """Maintain one fixed preference per batched environment episode."""

    def __init__(
        self,
        env: Env,
        logical_worker_ids: chex.Array,
        process_count: int = 10,
    ):
        super().__init__(env)
        _preference_subspace_bounds(process_count)
        self.logical_worker_ids = jnp.asarray(logical_worker_ids, dtype=jnp.int32)
        self.process_count = process_count

    def reset(self, key: chex.PRNGKey) -> EnvState:
        env_key, preference_key = jax.random.split(key)
        state = self.env.reset(env_key)
        if state.done.ndim != 1:
            raise ValueError("EpisodePreferenceWrapper requires a batched environment")

        keys = jax.random.split(preference_key, state.done.shape[0])
        keys, sample_keys = rng_split(keys)
        safe_worker_ids = jnp.maximum(self.logical_worker_ids, 0)
        preferences = jax.vmap(
            lambda sample_key, worker_id: sample_official_preference(
                sample_key, worker_id, self.process_count
            )
        )(sample_keys, safe_worker_ids)

        return state.replace(
            info=state.info.replace(
                preference=preferences,
                logical_worker_id=self.logical_worker_ids,
            ),
            _internal=state._internal.replace(preference_key=keys),
        )

    def step(self, state: EnvState, action: Action) -> EnvState:
        preferences = state.info.preference
        state = self.env.step(state, action)

        old_keys = state._internal.preference_key
        next_keys, sample_keys = rng_split(old_keys)
        safe_worker_ids = jnp.maximum(self.logical_worker_ids, 0)
        candidates = jax.vmap(
            lambda sample_key, worker_id: sample_official_preference(
                sample_key, worker_id, self.process_count
            )
        )(sample_keys, safe_worker_ids)
        done = state.done[..., None].astype(jnp.bool_)

        return state.replace(
            info=state.info.replace(
                preference=jnp.where(done, candidates, preferences)
            ),
            _internal=state._internal.replace(
                preference_key=jnp.where(done, next_keys, old_keys)
            ),
        )
