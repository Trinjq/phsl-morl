"""Validate a Brax key artifact through the production interpolation path."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from evorl.evaluators.pd_morl import load_key_solution_artifact, preference_grid
from evorl.utils.pd_morl_interpolator import (
    fit_interpolator_state,
    fit_reference_interpolator,
    interpolate,
    key_preferences,
    normalize_key_solutions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=Path("configs/artifacts/interp_objs_walker2d_brax.txt"),
    )
    args = parser.parse_args()

    keys = key_preferences(2)
    solutions, provenance = load_key_solution_artifact(args.artifact, keys)
    if np.argmax(solutions[:, 1]) != 0:
        raise AssertionError("[0, 1] key must have the largest objective 2")
    if np.argmax(solutions[:, 0]) != 2:
        raise AssertionError("[1, 0] key must have the largest objective 1")
    balanced = solutions[1]
    if np.any(
        np.all(solutions[[0, 2]] >= balanced, axis=1)
        & np.any(solutions[[0, 2]] > balanced, axis=1)
    ):
        raise AssertionError("balanced key is strictly dominated by an endpoint key")

    checks = {}
    queries = preference_grid(0.001)
    for normalization in ("initial", "online"):
        reference = fit_reference_interpolator(keys, solutions, normalization)
        state = fit_interpolator_state(keys, solutions, normalization)
        scipy_values = reference(queries)
        jax_values = np.asarray(jax.jit(interpolate)(state, queries))
        # Production interpolation is float32; SciPy evaluates the reference in
        # float64, so parity is checked at float32 numerical precision.
        np.testing.assert_allclose(jax_values, scipy_values, rtol=2e-3, atol=5e-4)
        expected_keys = normalize_key_solutions(solutions, normalization)
        np.testing.assert_allclose(reference(keys), expected_keys, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(interpolate(state, keys)), expected_keys, rtol=2e-5, atol=2e-5
        )
        if not np.isfinite(jax_values).all():
            raise AssertionError(f"{normalization} interpolation is non-finite")
        checks[normalization] = {
            "grid_min": float(jax_values.min()),
            "grid_max": float(jax_values.max()),
            "max_scipy_jax_abs_error": float(
                np.max(np.abs(jax_values - scipy_values))
            ),
        }

    print(
        json.dumps(
            {
                "artifact": provenance.__dict__,
                "row_order": keys.tolist(),
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
