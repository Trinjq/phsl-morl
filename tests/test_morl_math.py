import jax
import jax.numpy as jnp

from evorl.utils.morl_math import (
    cosine_similarity,
    directional_angle,
    normalize_vector,
    safe_norm,
    scalarize,
)


def test_scalarize_shapes_and_critic_axis_broadcast():
    assert scalarize(jnp.array([1.0, 2.0]), jnp.array([0.25, 0.75])) == 1.75

    q_vec = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    w = jnp.array([[0.25, 0.75], [0.5, 0.5]])
    assert jnp.allclose(scalarize(q_vec, w), jnp.array([1.75, 3.5]))

    twin_q = jnp.stack((q_vec, q_vec + 1), axis=1)
    result = scalarize(twin_q, w)
    assert result.shape == (2, 2)
    assert jnp.allclose(result, jnp.array([[1.75, 2.75], [3.5, 4.5]]))


def test_norm_normalize_and_zero_vector_behavior():
    eps = 1e-8
    zero = jnp.zeros(2, dtype=jnp.float32)
    y = jnp.array([1.0, 0.0], dtype=jnp.float32)

    assert safe_norm(zero) == eps
    assert jnp.array_equal(normalize_vector(zero), zero)
    assert cosine_similarity(zero, y) == 0
    assert jnp.allclose(directional_angle(zero, y), 90.0)
    assert jnp.all(jnp.isfinite(normalize_vector(zero)))
    assert jnp.isfinite(directional_angle(zero, y))
    zero_gradient = jax.grad(safe_norm)(zero)
    assert jnp.all(jnp.isfinite(zero_gradient))

    batched = jnp.array([[3.0, 4.0], [0.0, 0.0]], dtype=jnp.float32)
    assert jnp.allclose(safe_norm(batched), jnp.array([5.0, eps]))
    assert jnp.allclose(normalize_vector(batched), jnp.array([[0.6, 0.8], [0, 0]]))


def test_cosine_and_directional_angle_boundaries():
    x = jnp.array([1.0, 0.0], dtype=jnp.float32)
    vectors = jnp.array(
        [[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]],
        dtype=jnp.float32,
    )

    cosine = jax.vmap(cosine_similarity, in_axes=(None, 0))(x, vectors)
    angle = jax.vmap(directional_angle, in_axes=(None, 0))(x, vectors)
    upper_clamp_angle = jnp.rad2deg(jnp.arccos(jnp.array(0.9999)))
    assert jnp.allclose(
        cosine, jnp.array([1.0, 0.0, -1.0, 1 / jnp.sqrt(2.0)])
    )
    assert jnp.allclose(angle[0], upper_clamp_angle)
    assert jnp.allclose(angle[1:3], jnp.array([90.0, 90.0]))
    assert jnp.allclose(angle[3], 45.0)
    assert jnp.all(jnp.isfinite(angle))
    upper_gradient = jax.grad(lambda value: directional_angle(value, vectors[0]))(x)
    assert jnp.all(jnp.isfinite(upper_gradient))


def test_batched_geometry_float32_jit_and_vmap():
    x = jnp.array(
        [[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, -1.0]]],
        dtype=jnp.float32,
    )
    y = jnp.array(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [-1.0, 1.0]]],
        dtype=jnp.float32,
    )
    cosine = jax.jit(cosine_similarity)(x, y)
    vmapped = jax.vmap(cosine_similarity)(x, y)

    assert cosine.shape == (2, 2)
    assert cosine.dtype == jnp.float32
    assert jnp.allclose(cosine, vmapped)
    assert jnp.all(jnp.isfinite(jax.jit(directional_angle)(x, y)))


def test_directional_angle_has_finite_gradient_off_boundaries():
    y = jnp.array([0.2, 1.0], dtype=jnp.float32)
    gradient = jax.grad(lambda x: directional_angle(x, y))(
        jnp.array([1.0, 0.3], dtype=jnp.float32)
    )
    assert gradient.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(gradient))
