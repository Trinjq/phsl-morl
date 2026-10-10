import jax
import jax.numpy as jnp
import numpy as np
import pytest
from evorl.utils.pd_morl_interpolator import (
    fit_interpolator_state,
    interpolate,
    key_preferences,
    normalize_key_solutions,
    update_key_solution,
)
from scipy.interpolate import RBFInterpolator

KEYS = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
# Synthetic deterministic fixture, not official Walker key solutions.
SOLUTIONS = np.array([[120.0, 900.0], [550.0, 620.0], [980.0, 140.0]])
UPDATED_SOLUTIONS = np.array([[130.0, 920.0], [570.0, 650.0], [1010.0, 135.0]])
QUERIES = np.array(
    [[0.0, 1.0], [0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.8, 0.2], [1.0, 0.0]]
)


def _errors(reference, actual):
    absolute = np.abs(reference - actual)
    relative = absolute / np.maximum(np.abs(reference), np.finfo(reference.dtype).eps)
    return absolute.max(), absolute.mean(), relative.max()


@pytest.mark.parametrize(
    ("normalization", "solutions"),
    [("initial", SOLUTIONS), ("online", UPDATED_SOLUTIONS)],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_jax_golden(normalization, solutions, dtype):
    with jax.enable_x64(dtype == np.float64):
        reference = RBFInterpolator(
            KEYS,
            np.asarray(normalize_key_solutions(solutions, normalization)),
            kernel="linear",
            smoothing=0,
            degree=0,
        )(QUERIES)
        state = fit_interpolator_state(KEYS, solutions, normalization, dtype=dtype)
        queries = jnp.asarray(QUERIES, dtype=dtype)
        eager = np.asarray(interpolate(state, queries))
        compiled = np.asarray(jax.jit(interpolate)(state, queries))
        vmapped = np.asarray(jax.vmap(lambda w: interpolate(state, w))(queries))
        single_shape = interpolate(state, queries[0]).shape

    tolerance = 2e-5 if dtype == np.float32 else 1e-10
    assert _errors(reference, eager)[0] <= tolerance
    assert np.allclose(eager, compiled, rtol=0, atol=tolerance)
    assert np.allclose(eager, vmapped, rtol=0, atol=tolerance)
    assert eager.dtype == dtype
    assert single_shape == (2,)
    assert eager.shape == QUERIES.shape
    assert np.allclose(eager[[0, 3, 5]], np.asarray(state.normalized_key_solutions))


def test_official_walker_key_order_and_normalizations():
    assert np.array_equal(key_preferences(2), KEYS)
    assert np.allclose(
        np.linalg.norm(normalize_key_solutions(SOLUTIONS, "initial"), ord=2, axis=1), 1
    )
    assert np.allclose(
        np.linalg.norm(normalize_key_solutions(SOLUTIONS, "online"), ord=1, axis=1), 1
    )
    assert np.array_equal(
        normalize_key_solutions(np.zeros((1, 2)), "initial"), np.zeros((1, 2))
    )


def test_fit_is_jittable():
    fit = jax.jit(
        lambda solutions: fit_interpolator_state(
            jnp.asarray(KEYS), solutions, "online", dtype=jnp.float32
        )
    )
    state = fit(jnp.asarray(SOLUTIONS, dtype=jnp.float32))
    assert state.coefficients.shape == (4, 2)
    assert np.isfinite(np.asarray(state.coefficients)).all()


def test_update_rule_and_online_refit():
    kept, improved = update_key_solution([10.0, 20.0], [9.0, 100.0], [1.0, 0.0])
    assert not improved
    assert np.array_equal(kept, [10.0, 20.0])

    replaced, improved = update_key_solution([10.0, 20.0], [11.0, 0.0], [1.0, 0.0])
    assert improved
    assert np.array_equal(replaced, [11.0, 0.0])

    updated = SOLUTIONS.copy()
    updated[2] = replaced
    state = fit_interpolator_state(KEYS, updated, "online")
    assert np.allclose(
        np.asarray(interpolate(state, jnp.asarray(KEYS[2], dtype=jnp.float32))),
        normalize_key_solutions(updated, "online")[2],
        atol=2e-7,
    )
