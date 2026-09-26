"""Pure JAX mathematical primitives for multi-objective reinforcement learning."""

import jax.numpy as jnp


def scalarize(q_vec, w):
    """Return ``w^T q_vec`` with the objective dimension on the last axis."""
    if q_vec.ndim == w.ndim + 1:
        w = w[..., None, :]
    return jnp.sum(q_vec * w, axis=-1)


def safe_norm(x, axis=-1, keepdims=False, eps=1e-8):
    """Return a JAX-autodiff-safe L2 norm lower-bounded by ``eps``."""
    sq_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=keepdims)
    return jnp.sqrt(jnp.maximum(sq_sum, eps**2))


def normalize_vector(x, axis=-1, eps=1e-8):
    """Framework utility that safely L2-normalizes vectors."""
    return x / safe_norm(x, axis=axis, keepdims=True, eps=eps)


def cosine_similarity(x, y, axis=-1, eps=1e-8):
    """Return finite cosine similarity with PyTorch-style norm protection."""
    dot = jnp.sum(x * y, axis=axis)
    denominator = safe_norm(x, axis=axis, eps=eps) * safe_norm(
        y, axis=axis, eps=eps
    )
    return dot / denominator


def directional_angle(x, y, axis=-1, eps=1e-8):
    """Return the source-faithful PD-MORL directional angle in degrees."""
    cosine = cosine_similarity(x, y, axis=axis, eps=eps)
    cosine = jnp.clip(cosine, 0.0, 0.9999)
    return jnp.rad2deg(jnp.arccos(cosine))
