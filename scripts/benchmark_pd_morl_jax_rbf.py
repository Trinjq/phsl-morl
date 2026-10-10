"""Benchmark the fixed-shape PD-MORL RBF fit on the selected JAX device."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RBFInterpolator

from evorl.utils.pd_morl_interpolator import fit_interpolator_state, interpolate


def _ready(value):
    return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)


def _normalize(values, order):
    norms = np.linalg.norm(values, ord=order, axis=1, keepdims=True)
    return values / np.where(norms == 0, 1.0, norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refits", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    raw = np.loadtxt(args.artifact, delimiter=",").astype(np.float32)
    keys = np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
    solutions = jnp.asarray(raw)
    key_array = jnp.asarray(keys)
    queries = jnp.linspace(0, 1, 1001, dtype=jnp.float32)
    queries = jnp.stack((queries, 1 - queries), axis=-1)
    fit = jax.jit(lambda values: fit_interpolator_state(key_array, values, "online"))
    forward = jax.jit(lambda state: interpolate(state, queries))

    start = time.perf_counter()
    state = _ready(fit(solutions))
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(args.refits):
        state = _ready(fit(solutions))
    steady_refit_seconds = (time.perf_counter() - start) / args.refits
    start = time.perf_counter()
    grid = _ready(forward(state))
    grid_seconds = time.perf_counter() - start

    scipy_times = {}
    for name, order in (("scipy_l1", 1), ("scipy_l2", 2)):
        targets = _normalize(raw, order)
        start = time.perf_counter()
        for _ in range(args.refits):
            RBFInterpolator(keys, targets, kernel="linear", smoothing=0, degree=0)
        scipy_times[name] = (time.perf_counter() - start) / args.refits

    result = {
        "backend": "jax",
        "normalization": "l1",
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "artifact": str(args.artifact),
        "artifact_sha256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
        "refits": args.refits,
        "compile_seconds": compile_seconds,
        "steady_refit_seconds": steady_refit_seconds,
        "grid_1001_seconds": grid_seconds,
        "scipy_reference_seconds": scipy_times,
        "grid_finite": bool(np.isfinite(np.asarray(grid)).all()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
