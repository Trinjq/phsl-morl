# MORL math primitives

`evorl/utils/morl_math.py` contains the framework-independent, pure-JAX
geometry used by MORL code. It has no NumPy, SciPy, or PyTorch runtime
dependency.

| Function | Definition | Output |
| --- | --- | --- |
| `scalarize(q_vec, w)` | `sum(q_vec * w, axis=-1)` | weighted scalar value |
| `safe_norm(x)` | `sqrt(max(sum(x**2), eps**2))` | AD-safe L2 norm bounded below by `eps` |
| `normalize_vector(x)` | `x / safe_norm(x, keepdims=True)` | normalized vector |
| `cosine_similarity(x, y)` | `(x dot y) / (max(norm(x), eps) * max(norm(y), eps))` | cosine |
| `directional_angle(x, y)` | `rad2deg(acos(clip(cosine, 0, 0.9999)))` | angle in degrees |

## Shapes and numerical behavior

The vector/objective axis defaults to the last axis. The functions support
`[L]`, `[B,L]`, and `[B,C,L]` inputs. `scalarize` also broadcasts `[B,L]`
preferences across the critic axis of `[B,C,L]` Q-values, preserving the
Step 4.4 MO-TD3 behavior.

`eps` defaults to `1e-8`. `safe_norm` clamps the squared sum to `eps**2`
before the square root. This **FRAMEWORK-ADAPTATION** preserves the forward
value while preventing a non-finite JAX gradient at the zero vector. A zero
vector therefore has safe norm `eps`, remains zero after normalization, has
cosine zero against any vector, and has directional angle 90 degrees.

`directional_angle` is the **SOURCE-FAITHFUL** PD-MORL definition:

```text
angle_deg = rad2deg(acos(clip(cosine_similarity(x, y), 0.0, 0.9999)))
```

Its unit is degrees. The theoretical output range is approximately 0.81
degrees (`rad2deg(acos(0.9999))`) through 90 degrees. Negative cosine values
are clamped to zero, so they all produce 90 degrees. The upper clamp also
avoids the singular `acos(1)` gradient.

All primitives run with float32 under `jax.jit` and `jax.vmap`.

`scalarize`, `cosine_similarity`, and `directional_angle` are
**SOURCE-FAITHFUL** algorithm-facing primitives. `safe_norm` and
`normalize_vector` are framework utilities. `normalize_vector` must not be
automatically applied to preference `w`, interpolator output `w_p`, or Q
vectors; in particular, it must not add an extra L2 normalization to `w_p`.

## Refactor and verification

The former `scalarize` implementation in `evorl/algorithms/mo_td3.py` was
moved here; MO-TD3 imports the single shared implementation. No actor loss,
critic loss, target, HER, interpolator, projected-preference, or evaluation
behavior was added or changed.

Verification covers hand-computed and batched scalarization, critic-axis
broadcasting, norm and zero-vector boundaries, same/orthogonal/opposite
directions, degrees, `[B,C,L]`, float32, JIT, vmap, and finite-gradient checks
test. `tests/test_mo_td3.py` is rerun as the Step 4.4 regression check.

- GPU 2, MORL math: `5 passed`.
- GPU 2, math plus fast MO-TD3 regression: `9 passed, 1 deselected`.
- GPU 0, complete MO-TD3 regression including Brax Walker: `5 passed`.
