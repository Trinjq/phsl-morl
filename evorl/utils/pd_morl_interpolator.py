"""PD-MORL's pure-JAX linear RBF interpolator."""

from typing import Literal

import jax.numpy as jnp
import numpy as np

from evorl.types import PyTreeData

Normalization = Literal["initial", "online"]


class PDMORLInterpolatorState(PyTreeData):
    """Fixed-shape arrays needed to evaluate a degree-zero linear RBF fit."""

    key_preferences: jnp.ndarray
    normalized_key_solutions: jnp.ndarray
    coefficients: jnp.ndarray
    shift: jnp.ndarray
    scale: jnp.ndarray
    powers: jnp.ndarray
    epsilon: jnp.ndarray


def key_preferences(num_objectives: int) -> np.ndarray:
    """Return one-hot keys plus the uniform key in official Walker order."""
    if num_objectives < 2:
        raise ValueError("num_objectives must be at least 2")
    keys = np.vstack(
        (np.eye(num_objectives), np.full(num_objectives, 1 / num_objectives))
    )
    return np.unique(keys, axis=0)


def normalize_key_solutions(
    key_solutions: jnp.ndarray, normalization: Normalization
) -> jnp.ndarray:
    """Normalize each raw key return with the objective-vector L2 norm."""
    if normalization not in ("initial", "online"):
        raise ValueError("normalization must be 'initial' or 'online'")
    solutions = jnp.asarray(key_solutions)
    if solutions.ndim != 2:
        raise ValueError("key_solutions must have shape [K, L]")
    norms = jnp.linalg.norm(
        solutions, ord=2 if normalization == "initial" else 1, axis=1, keepdims=True
    )
    return solutions / jnp.where(norms > 0, norms, 1)


def fit_interpolator_state(
    keys: jnp.ndarray,
    key_solutions: jnp.ndarray,
    normalization: Normalization,
    *,
    dtype=jnp.float32,
) -> PDMORLInterpolatorState:
    """Fit linear, smoothing-zero, degree-zero RBF system in JAX."""
    keys = jnp.asarray(keys, dtype=dtype)
    key_solutions = jnp.asarray(key_solutions, dtype=dtype)
    if keys.ndim != 2 or key_solutions.ndim != 2 or keys.shape != key_solutions.shape:
        raise ValueError("keys and key_solutions must share shape [K, L]")

    targets = normalize_key_solutions(key_solutions, normalization)
    distances = jnp.linalg.norm(keys[:, None, :] - keys[None, :, :], axis=-1)
    phi = -distances
    polynomial = jnp.ones((keys.shape[0], 1), dtype=dtype)
    zero = jnp.zeros((1, 1), dtype=dtype)
    system = jnp.concatenate(
        (
            jnp.concatenate((phi, polynomial), axis=1),
            jnp.concatenate((polynomial.T, zero), axis=1),
        ),
        axis=0,
    )
    rhs = jnp.concatenate(
        (targets, jnp.zeros((1, key_solutions.shape[1]), dtype=dtype)), axis=0
    )
    coefficients = jnp.linalg.solve(system, rhs)
    return PDMORLInterpolatorState(
        key_preferences=keys,
        normalized_key_solutions=targets,
        coefficients=coefficients,
        shift=jnp.zeros((keys.shape[1],), dtype=dtype),
        scale=jnp.ones((keys.shape[1],), dtype=dtype),
        powers=jnp.zeros((1, keys.shape[1]), dtype=jnp.int32),
        epsilon=jnp.asarray(1, dtype=dtype),
    )


def interpolate(state: PDMORLInterpolatorState, preference: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the interpolator for one [L] input or a batch [..., L]."""
    preference = jnp.asarray(preference)
    radial = -state.epsilon * jnp.linalg.norm(
        preference[..., None, :] - state.key_preferences, axis=-1
    )
    scaled = (preference - state.shift) / state.scale
    polynomial = jnp.prod(scaled[..., None, :] ** state.powers, axis=-1)
    basis = jnp.concatenate((radial, polynomial), axis=-1)
    return basis @ state.coefficients


def update_key_solution(
    old_solution: np.ndarray,
    candidate_solution: np.ndarray,
    key_preference: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Replace a key solution iff its official scalarized score improves."""
    old = np.asarray(old_solution)
    candidate = np.asarray(candidate_solution)
    preference = np.asarray(key_preference)
    if old.shape != candidate.shape or old.shape != preference.shape:
        raise ValueError("solution and preference vectors must share shape [L]")
    improved = bool(preference @ candidate > preference @ old)
    return (candidate.copy() if improved else old.copy()), improved
