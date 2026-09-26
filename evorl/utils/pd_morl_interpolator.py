"""PD-MORL's host-fitted linear RBF interpolator with a pure JAX forward."""

from typing import Literal

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RBFInterpolator

from evorl.types import PyTreeData

Normalization = Literal["initial", "online"]


class PDMORLInterpolatorState(PyTreeData):
    """Fixed-shape arrays needed to evaluate SciPy's linear RBF fit."""

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
    key_solutions: np.ndarray, normalization: Normalization
) -> np.ndarray:
    """Apply the source's initial L2 or online-update L1 normalization."""
    solutions = np.asarray(key_solutions, dtype=np.float64)
    if solutions.ndim != 2:
        raise ValueError("key_solutions must have shape [K, L]")
    order = (
        2 if normalization == "initial" else 1 if normalization == "online" else None
    )
    if order is None:
        raise ValueError("normalization must be 'initial' or 'online'")
    norms = np.linalg.norm(solutions, ord=order, axis=1, keepdims=True)
    # ponytail: preserve zero fixtures; reject/replace them if training artifacts need it.
    return solutions / np.where(norms == 0, 1.0, norms)


def fit_reference_interpolator(
    keys: np.ndarray,
    key_solutions: np.ndarray,
    normalization: Normalization,
) -> RBFInterpolator:
    """Fit the exact host-side interpolator used by the official source."""
    keys = np.asarray(keys, dtype=np.float64)
    targets = normalize_key_solutions(key_solutions, normalization)
    if keys.ndim != 2 or keys.shape != targets.shape:
        raise ValueError("keys and key_solutions must share shape [K, L]")
    return RBFInterpolator(keys, targets, kernel="linear")


def fit_interpolator_state(
    keys: np.ndarray,
    key_solutions: np.ndarray,
    normalization: Normalization,
    *,
    dtype=np.float32,
) -> PDMORLInterpolatorState:
    """Fit on the host and copy only fixed-shape numerical state to JAX."""
    reference = fit_reference_interpolator(keys, key_solutions, normalization)
    to_jax = lambda value: jnp.asarray(value, dtype=dtype)
    return PDMORLInterpolatorState(
        key_preferences=to_jax(reference.y),
        normalized_key_solutions=to_jax(reference.d),
        coefficients=to_jax(reference._coeffs),
        shift=to_jax(reference._shift),
        scale=to_jax(reference._scale),
        powers=jnp.asarray(reference.powers, dtype=jnp.int32),
        epsilon=to_jax(reference.epsilon),
    )


def interpolate(state: PDMORLInterpolatorState, preference: jnp.ndarray) -> jnp.ndarray:
    """Evaluate ``I(w)`` for one ``[L]`` input or a batch ``[..., L]``."""
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
