"""Host-side PD-MORL control and evaluation behavior."""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pymoo.indicators.hv import HV

from evorl.agent import AgentState
from evorl.rollout import fast_eval_rollout_episode
from evorl.sample_batch import SampleBatch
from evorl.types import PyTreeDict
from evorl.utils.pd_morl_interpolator import (
    PDMORLInterpolatorState,
    fit_interpolator_state,
    key_preferences,
)


@dataclass(frozen=True)
class KeySolutionArtifact:
    path: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str
    key_preferences: list[list[float]]
    raw_values: list[list[float]]


@dataclass(frozen=True)
class MORLEvaluationResult:
    preferences: np.ndarray
    returns_per_repeat: np.ndarray
    mean_returns: np.ndarray
    hv_per_repeat: np.ndarray
    sparsity_per_repeat: np.ndarray
    mean_hv: float
    mean_sparsity: float
    pareto_indices: np.ndarray
    pareto_returns: np.ndarray
    pareto_counts_per_repeat: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    mean_pareto_count: float = 0.0
    source_hv: float = 0.0
    source_sparsity: float = 0.0
    source_pareto_point_count: int = 0
    mean_repeat_hv: float = 0.0
    mean_repeat_sparsity: float = 0.0
    mean_repeat_pareto_count: float = 0.0


def load_key_solution_artifact(
    path: str | Path, keys: np.ndarray
) -> tuple[np.ndarray, KeySolutionArtifact]:
    """Load required key solutions and record their complete provenance."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"interpolator artifact does not exist: {path}")

    keys = np.asarray(keys, dtype=np.float64)
    if path.suffix == ".npz":
        with np.load(path) as artifact:
            if "key_solutions" not in artifact:
                raise ValueError(".npz artifact must contain 'key_solutions'")
            solutions = np.asarray(artifact["key_solutions"])
            if "key_preferences" in artifact and not np.array_equal(
                artifact["key_preferences"], keys
            ):
                raise ValueError("artifact key preference ordering does not match")
    else:
        solutions = np.loadtxt(path, delimiter=",")

    if solutions.shape != keys.shape:
        raise ValueError(
            f"key solutions must have shape {keys.shape}, got {solutions.shape}"
        )
    if not np.isfinite(solutions).all():
        raise ValueError("key solutions must be finite")

    artifact_bytes = path.read_bytes()
    if path.suffix != ".npz":
        artifact_bytes = artifact_bytes.replace(b"\r\n", b"\n")
    provenance = KeySolutionArtifact(
        path=str(path),
        sha256=sha256(artifact_bytes).hexdigest(),
        shape=solutions.shape,
        dtype=str(solutions.dtype),
        key_preferences=keys.tolist(),
        raw_values=solutions.tolist(),
    )
    return solutions, provenance


def evaluation_seeds(repeats: int) -> np.ndarray:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return np.arange(repeats, dtype=np.int64) * 11


def preference_grid(step: float) -> np.ndarray:
    size = round(1.0 / step)
    if step <= 0 or not np.isclose(size * step, 1.0):
        raise ValueError("step must divide one exactly")
    first = np.arange(size + 1, dtype=np.float64) / size
    return np.stack((first, first[::-1]), axis=-1)


def key_update_due(episode_count: np.ndarray, eval_cnt_ep: int) -> bool:
    return bool(np.all(np.asarray(episode_count) > eval_cnt_ep))


def full_evaluation_due(
    episode_count: np.ndarray, eval_cnt: int, eval_freq: int = 100
) -> bool:
    return bool(np.all(np.asarray(episode_count) > eval_freq * eval_cnt))


def aggregate_candidate_returns(returns_per_repeat: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns_per_repeat)
    if returns.ndim != 3:
        raise ValueError("returns_per_repeat must have shape [S, K, L]")
    return returns.mean(axis=0)


def replace_key_solutions(
    old_solutions: np.ndarray,
    candidate_solutions: np.ndarray,
    keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    old = np.asarray(old_solutions)
    candidates = np.asarray(candidate_solutions)
    keys = np.asarray(keys)
    if old.shape != candidates.shape or old.shape != keys.shape:
        raise ValueError("old, candidate, and key arrays must share shape [K, L]")
    improved = np.sum(keys * candidates, axis=1) > np.sum(keys * old, axis=1)
    return np.where(improved[:, None], candidates, old).copy(), improved


def update_key_solutions(
    old_solutions: np.ndarray,
    returns_per_repeat: np.ndarray,
    keys: np.ndarray,
    *,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, PDMORLInterpolatorState]:
    candidates = aggregate_candidate_returns(returns_per_repeat)
    updated, improved = replace_key_solutions(old_solutions, candidates, keys)
    # Source refits after every trigger, including when no key was replaced.
    state = fit_interpolator_state(keys, updated, "online", dtype=dtype)
    return updated, improved, state


def _update_key_solutions_jax(
    old_solutions: jax.Array,
    returns_per_repeat: jax.Array,
    keys: jax.Array,
) -> tuple[jax.Array, jax.Array, PDMORLInterpolatorState]:
    old = jnp.asarray(old_solutions)
    candidates = jnp.asarray(returns_per_repeat).mean(axis=0)
    keys = jnp.asarray(keys, dtype=old.dtype)
    improved = jnp.sum(keys * candidates, axis=1) > jnp.sum(keys * old, axis=1)
    updated = jnp.where(improved[:, None], candidates, old)
    state = fit_interpolator_state(keys, updated, "online", dtype=old.dtype)
    return updated, improved, state


update_key_solutions_jax = jax.jit(_update_key_solutions_jax)


def non_dominated_indices(returns: np.ndarray) -> np.ndarray:
    """Maximization non-dominated front, retaining duplicate entries."""
    points = np.asarray(returns)
    if points.ndim != 2:
        raise ValueError("returns must have shape [N, L]")
    dominated = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        dominated[i] = np.any(
            np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        )
    return np.flatnonzero(~dominated)


def hypervolume(returns: np.ndarray) -> float:
    """Official pymoo minimization adapter with zero reference."""
    points = np.asarray(returns, dtype=np.float64)
    return float(HV(ref_point=np.zeros(points.shape[1]))(-points))


def sparsity(returns: np.ndarray) -> float:
    points = np.asarray(returns, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("returns must have shape [N, L]")
    points = points[non_dominated_indices(points)]
    if len(points) <= 1:
        return 0.0
    ordered = np.sort(points, axis=0)
    return float(np.square(np.diff(ordered, axis=0)).sum() / (len(points) - 1))


def evaluate_morl(
    evaluate_episode: Callable[[np.ndarray, int, int], np.ndarray],
    preferences: np.ndarray,
    repeats: int,
) -> MORLEvaluationResult:
    """Evaluate in the official repeat-then-preference logical order."""
    preferences = np.asarray(preferences, dtype=np.float64)
    rows = [
        [
            evaluate_episode(preference, int(seed), i)
            for i, preference in enumerate(preferences)
        ]
        for seed in evaluation_seeds(repeats)
    ]
    return morl_evaluation_result(preferences, np.asarray(rows))


def morl_evaluation_result(
    preferences: np.ndarray, returns_per_repeat: np.ndarray
) -> MORLEvaluationResult:
    """Build the shared serial/batched MORL result on the host."""
    preferences = np.asarray(preferences, dtype=np.float64)
    returns = np.asarray(returns_per_repeat)
    if returns.ndim != 3 or returns.shape[1] != len(preferences):
        raise ValueError("returns_per_repeat must have shape [S, K, L]")
    hv_list = []
    sp_list = []
    counts = []
    for row in returns:
        idx = non_dominated_indices(row)
        front = row[idx]
        hv_list.append(hypervolume(front))
        sp_list.append(sparsity(front))
        counts.append(len(front))
    hv = np.asarray(hv_list, dtype=np.float64)
    sp = np.asarray(sp_list, dtype=np.float64)
    pareto_counts = np.asarray(counts, dtype=np.int64)
    mean_returns = returns.mean(axis=0)
    indices = non_dominated_indices(mean_returns)
    source_pareto = mean_returns[indices]
    source_hv = hypervolume(source_pareto)
    source_sp = sparsity(source_pareto)
    source_count = len(source_pareto)
    mean_rep_hv = float(hv.mean())
    mean_rep_sp = float(sp.mean())
    mean_rep_cnt = float(pareto_counts.mean())

    return MORLEvaluationResult(
        preferences=preferences,
        returns_per_repeat=returns,
        mean_returns=mean_returns,
        hv_per_repeat=hv,
        sparsity_per_repeat=sp,
        mean_hv=mean_rep_hv,
        mean_sparsity=mean_rep_sp,
        pareto_indices=indices,
        pareto_returns=source_pareto,
        pareto_counts_per_repeat=pareto_counts,
        mean_pareto_count=mean_rep_cnt,
        source_hv=source_hv,
        source_sparsity=source_sp,
        source_pareto_point_count=source_count,
        mean_repeat_hv=mean_rep_hv,
        mean_repeat_sparsity=mean_rep_sp,
        mean_repeat_pareto_count=mean_rep_cnt,
    )


class PDMORLEvaluator:
    """Serial logical evaluator over a one-lane JAX/Brax environment."""

    def __init__(self, env, agent, max_episode_steps: int):
        if env.num_envs != 1:
            raise ValueError(
                "source-comparison evaluation requires one environment lane"
            )
        self.env = env
        self.agent = agent
        self.max_episode_steps = max_episode_steps
        self._episode = jax.jit(self._evaluate_episode)

    def _evaluate_episode(
        self,
        agent_state: AgentState,
        preference: jax.Array,
        seed: jax.Array,
        preference_index: jax.Array,
    ) -> jax.Array:
        # FRAMEWORK-ADAPTATION: fold the serial reset position into the source seed.
        reset_key = jax.random.fold_in(jax.random.PRNGKey(seed), preference_index)
        rollout_key = jax.random.fold_in(reset_key, 1)
        env_state = self.env.reset(reset_key)

        def action_fn(current_agent_state, batch, key):
            batch = SampleBatch(
                obs=batch.obs,
                extras=PyTreeDict(
                    policy_extras=PyTreeDict(
                        preference=jnp.broadcast_to(
                            preference, (*batch.obs.shape[:-1], preference.shape[-1])
                        )
                    )
                ),
            )
            return self.agent.evaluate_actions(current_agent_state, batch, key)

        metrics, _ = fast_eval_rollout_episode(
            self.env.step,
            action_fn,
            env_state,
            agent_state,
            rollout_key,
            self.max_episode_steps,
        )
        return metrics.episode_returns[0]

    def evaluate(
        self, agent_state: AgentState, preferences: np.ndarray, repeats: int
    ) -> MORLEvaluationResult:
        return evaluate_morl(
            lambda preference, seed, index: np.asarray(
                self._episode(agent_state, preference, seed, index)
            ),
            preferences,
            repeats,
        )

    def evaluate_keys_device(
        self, agent_state: AgentState, repeats: int = 3
    ) -> jax.Array:
        """Return key returns through the evaluator interface used by the workflow."""
        result = self.evaluate(agent_state, key_preferences(2), repeats)
        return jnp.asarray(result.returns_per_repeat)


class BatchedPDMORLEvaluator:
    """GPU-native evaluator with fixed-shape key and full-evaluation batches."""

    def __init__(self, key_env, env, agent, max_episode_steps: int):
        if key_env.num_envs != 9:
            raise ValueError("key evaluation requires exactly 9 environment lanes")
        if env.num_envs < 1:
            raise ValueError("eval batch size must be positive")
        self.key_env = key_env
        self.env = env
        self.agent = agent
        self.max_episode_steps = max_episode_steps
        self._key_batch = jax.jit(lambda *args: self._evaluate_batch(key_env, *args))
        self._full_batch = jax.jit(lambda *args: self._evaluate_batch(env, *args))

    def _evaluate_batch(
        self,
        env,
        agent_state: AgentState,
        preferences: jax.Array,
        seeds: jax.Array,
        preference_indices: jax.Array,
    ) -> jax.Array:
        reset_keys = jax.vmap(
            lambda seed, index: jax.random.fold_in(
                jax.random.PRNGKey(seed), index
            )
        )(seeds, preference_indices)
        rollout_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(reset_keys)
        # VmapWrapper(num_envs=1) splits the serial reset key once. Reproduce
        # that lane key so serial and batched evaluation see identical resets.
        lane_reset_keys = jax.vmap(lambda key: jax.random.split(key, 1)[0])(
            reset_keys
        )
        env_state = env.reset(lane_reset_keys)

        def action_fn(current_agent_state, batch, key):
            batch = SampleBatch(
                obs=batch.obs,
                extras=PyTreeDict(
                    policy_extras=PyTreeDict(preference=preferences)
                ),
            )
            return self.agent.evaluate_actions(current_agent_state, batch, key)

        metrics, _ = fast_eval_rollout_episode(
            env.step,
            action_fn,
            env_state,
            agent_state,
            rollout_keys,
            self.max_episode_steps,
        )
        return metrics.episode_returns

    def evaluate(
        self, agent_state: AgentState, preferences: np.ndarray, repeats: int
    ) -> MORLEvaluationResult:
        preferences = np.asarray(preferences, dtype=np.float64)
        if preferences.ndim != 2 or len(preferences) == 0:
            raise ValueError("preferences must have shape [K, L] with K > 0")

        seeds = evaluation_seeds(repeats)
        count = len(preferences)
        flat_preferences = np.tile(preferences, (repeats, 1))
        flat_seeds = np.repeat(seeds, count)
        flat_indices = np.tile(np.arange(count, dtype=np.int64), repeats)

        if len(flat_preferences) == 9:
            flat_returns = np.asarray(
                self._key_batch(
                    agent_state, flat_preferences, flat_seeds, flat_indices
                )
            )
        else:
            batch_size = self.env.num_envs
            chunks = []
            for start in range(0, len(flat_preferences), batch_size):
                stop = min(start + batch_size, len(flat_preferences))
                valid = stop - start
                batch_preferences = np.zeros(
                    (batch_size, preferences.shape[1]), dtype=preferences.dtype
                )
                batch_seeds = np.zeros((batch_size,), dtype=flat_seeds.dtype)
                batch_indices = np.zeros((batch_size,), dtype=flat_indices.dtype)
                batch_preferences[:valid] = flat_preferences[start:stop]
                batch_seeds[:valid] = flat_seeds[start:stop]
                batch_indices[:valid] = flat_indices[start:stop]
                chunks.append(
                    np.asarray(
                        self._full_batch(
                            agent_state,
                            batch_preferences,
                            batch_seeds,
                            batch_indices,
                        )
                    )[:valid]
                )
            flat_returns = np.concatenate(chunks, axis=0)

        returns = flat_returns.reshape(repeats, count, -1)
        return morl_evaluation_result(preferences, returns)

    def evaluate_keys_device(
        self, agent_state: AgentState, repeats: int = 3
    ) -> jax.Array:
        """Return fixed-shape key returns without crossing the host boundary."""
        if repeats != 3:
            raise ValueError("GPU-native key evaluation requires exactly 3 repeats")
        preferences = jnp.asarray(key_preferences(2), dtype=jnp.float32)
        count = preferences.shape[0]
        flat_preferences = jnp.tile(preferences, (repeats, 1))
        flat_seeds = jnp.repeat(jnp.asarray(evaluation_seeds(repeats)), count)
        flat_indices = jnp.tile(jnp.arange(count, dtype=jnp.int32), repeats)
        returns = self._key_batch(
            agent_state, flat_preferences, flat_seeds, flat_indices
        )
        return returns.reshape(repeats, count, -1)
