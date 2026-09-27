"""Host-side PD-MORL control and evaluation behavior."""

from collections.abc import Callable
from dataclasses import dataclass
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

    provenance = KeySolutionArtifact(
        path=str(path),
        sha256=sha256(path.read_bytes()).hexdigest(),
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
    returns = np.asarray(rows)
    hv = np.asarray([hypervolume(row) for row in returns])
    sp = np.asarray([sparsity(row) for row in returns])
    mean_returns = returns.mean(axis=0)
    indices = non_dominated_indices(mean_returns)
    return MORLEvaluationResult(
        preferences=preferences,
        returns_per_repeat=returns,
        mean_returns=mean_returns,
        hv_per_repeat=hv,
        sparsity_per_repeat=sp,
        mean_hv=float(hv.mean()),
        mean_sparsity=float(sp.mean()),
        pareto_indices=indices,
        pareto_returns=mean_returns[indices],
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


class KeyInterpolatorUpdateController:
    """Source-faithful host controller for key evaluation and online refit."""

    def __init__(self, objective_size: int, repeats: int = 3):
        if objective_size != 2:
            raise ValueError("the frozen PD-MORL baseline requires two objectives")
        self.keys = key_preferences(objective_size)
        self.repeats = repeats

    def update(self, evaluator, agent_state, raw_key_solutions):
        result = evaluator.evaluate(agent_state, self.keys, repeats=self.repeats)
        return update_key_solutions(
            np.asarray(jax.device_get(raw_key_solutions)),
            result.returns_per_repeat,
            self.keys,
        )
