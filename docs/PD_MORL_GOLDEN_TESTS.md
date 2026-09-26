# PD-MORL PyTorch ↔ JAX/EvoRL Golden Tests

## 1. Purpose and conclusion

This audit isolates PD-MORL numerical semantics from environment physics and
stochastic training. Official PyTorch operations and EvoRL/JAX operations use
the same synthetic batch, canonical parameters, target noise, and interpolator
fixture. CPU float64/float32, GPU float64, and the explicit GPU float32
reproduction policy all pass the frozen tolerance.

**STEP6 GOLDEN TEST PASS WITH EXPLICIT GPU PRECISION POLICY.**

## 2. Reference and provenance

- Official PD-MORL commit: `35aa1bccb31c149c3f3bb895c7c6ffa674e4b461`.
- EvoRL export base commit: `c1a1432340d71425a372ebef18afcd3eb5390d5e`.
- Audited official files: `lib/models/networks.py`,
  `lib/common_ptan/agent.py`, `PD-MORL/train_Walker2d_MO_TD3_HER.py`, and
  `lib/utilities/settings.py`.
- Full runtime versions and constants are recorded in
  `tests/golden/pd_morl_golden_metadata.json`.

The interpolator points are explicitly a **TEST FIXTURE**, not an official
Walker Pareto result.

## 3. Canonical fixture

`tests/golden/pd_morl_walker_input.npz` is generated with NumPy seed
`20260306`. It contains `B=4`, Walker dimensions `(obs=17, action=6,
objectives=2)`, states, actions, vector rewards, next states, done flags,
preferences, fixed target noise, all online/target network weights and biases,
and interpolator inputs/state sources. Constants are `gamma=0.995`,
`actor_loss_coeff=10`, and `policy_freq=10`.

The export command is:

```text
python tests/golden/export_pd_morl_pytorch_reference.py
```

It writes both the fixture and
`tests/golden/pd_morl_pytorch_reference.npz`. Random numbers are never sampled
inside either framework comparison.

## 4. Parameter and operation mapping

| PyTorch official | JAX/EvoRL | Mapping |
| --- | --- | --- |
| `Linear.weight [out,in]` | `Dense.kernel [in,out]` | exact transpose |
| `Linear.bias [out]` | `Dense.bias [out]` | unchanged |
| Actor input | Actor input | `concat(state, w)` |
| Actor layers | `MLP` layers | `400-ReLU-400-ReLU-6-tanh`, then action scale |
| Critic input | Critic input | `concat(state, w, action)` |
| `Q1`, `Q2` | `critic_0`, `critic_1` | independent parameters, output `[B,2]` each |
| objective order | objective order | Walker forward, healthy |
| scalar objective | `scalarize` | original `w`, never `wp` |
| angle objective | `directional_angle` | interpolated `wp` and vector Q |
| target twin minimum | `select_pessimistic_q_vector` | lower `wᵀQ`; select whole vector |

`torch_to_jax_params` and `torch_to_jax_critic_params` make this mapping
explicit. The parameter round-trip test has maximum absolute error exactly
zero after the requested dtype cast.

## 5. Compared calculations

The PyTorch export mirrors the official network order and source formulas. The
tests compare actor and target-actor forward passes; current and target Q1/Q2;
`wᵀQ1` and `wᵀQ2`; SciPy/JAX interpolator output; cosine and directional angle;
whole-vector target critic selection; vector Bellman target; both mean
Smooth-L1 terms; both unweighted critic angle terms; critic total loss; actor
Q1 scalarization; actor angle; and
`-mean(wᵀQ1) + 10 mean(angle)`.

The angle boundary test confirms same-direction and different-magnitude inputs
give `0.810291°`, while orthogonal and opposite inputs give `90°`, matching the
official `clip(cos, 0, 0.9999)` behavior.

## 6. Precision Policy

The official PD-MORL experiments used PyTorch on CPU and therefore define no
official GPU matrix-multiplication precision policy. The following distinction
is frozen for this reproduction:

- **SOURCE-FAITHFUL:** PD-MORL formulas, network order, vector-Q behavior,
  scalarization, targets, interpolator, angles, losses, and raw gradients.
- **FRAMEWORK-ADAPTATION:** JAX GPU float32 matrix multiplication uses
  `highest` precision to preserve the PyTorch/CPU reference semantics.

The production configuration declares `matmul_precision: highest`, and
`MOTD3Workflow._build_from_config` applies it through
`jax_default_matmul_precision` before environment, network, or learner setup.
This does not alter an algorithm formula.

The retained diagnostics show:

- JAX GPU default float32: FAIL under the frozen tolerance;
- JAX GPU float64: PASS;
- JAX GPU float32 with `highest`: PASS (`28 passed`);
- tolerance values were not modified.

Both runs are preserved in
`tests/golden/pd_morl_gpu_precision_diagnostic.json` so JAX backend defaults
cannot be confused with the PD-MORL reproduction policy.

## 7. Frozen tolerances

Tolerances were frozen from the corrected CPU reference comparison before the
GPU run. They were not enlarged after the default GPU float32 failure.

| Category | dtype | atol | rtol |
| --- | ---: | ---: | ---: |
| exact/scalarization | float64 | 2e-14 | 2e-14 |
| forward/interpolator | float64 | 2e-13 | 2e-13 |
| angle/loss | float64 | 5e-12 | 5e-12 |
| raw gradient | float64 | 5e-12 | 1e-9 |
| exact/scalarization | float32 | 2e-6 | 2e-6 |
| forward/interpolator | float32 | 2e-6 | 1e-5 |
| angle/loss | float32 | 2e-4 | 2e-6 |
| raw gradient | float32 | 1e-4 | 1e-4 |

## 8. CPU measured numerical errors

For tensors, the PyTorch/JAX columns report the shared shape. Scalar rows show
`PyTorch / JAX`. All CPU rows passed their category-specific tolerance above;
they are complemented by the GPU policy results in sections 6 and 10.

| Metric | dtype | PyTorch / JAX | Max abs | Mean abs | Max rel | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| actor output | f64 | `(4,6)` | 3.33e-16 | 1.10e-16 | 5.57e-15 | PASS |
| critic Q1 | f64 | `(4,2)` | 2.08e-16 | 7.85e-17 | 1.85e-15 | PASS |
| critic Q2 | f64 | `(4,2)` | 3.33e-16 | 1.55e-16 | 8.14e-15 | PASS |
| scalar Q1/Q2 | f64 | `(4,)` | 1.46e-16 | 5.72e-17 | 3.43e-14 | PASS |
| interpolator `wp` | f64 | `(4,2)` | 1.11e-16 | 4.16e-17 | 1.96e-16 | PASS |
| cosine Q1/Q2 | f64 | `(4,)` | 7.77e-16 | 4.93e-16 | 2.56e-15 | PASS |
| angle Q1/Q2 | f64 | `(4,)` | 7.11e-14 | 3.73e-14 | 1.67e-15 | PASS |
| selected target Q | f64 | `(4,2)` | 1.94e-16 | 5.64e-17 | 1.15e-15 | PASS |
| TD target | f64 | `(4,2)` | 1.11e-16 | 1.39e-17 | 4.33e-16 | PASS |
| critic total | f64 | `127.763289 / 127.763289` | 1.42e-14 | 1.42e-14 | 1.11e-16 | PASS |
| actor total | f64 | `654.556423 / 654.556423` | 1.14e-13 | 1.14e-13 | 1.74e-16 | PASS |
| critic gradients | f64 | all layers | 8.53e-14 | 6.41e-16 | 8.04e-11 | PASS |
| actor gradients | f64 | all layers | 2.98e-13 | 2.04e-15 | 6.10e-10 | PASS |
| actor output | f32 | `(4,6)` | 1.79e-7 | 4.40e-8 | 1.79e-6 | PASS |
| critic Q1/Q2 | f32 | `(4,2)` | 1.04e-7 | 4.94e-8 | 7.00e-6 | PASS |
| scalar Q1/Q2 | f32 | `(4,)` | 5.96e-8 | 3.35e-8 | 1.40e-5 | PASS |
| interpolator `wp` | f32 | `(4,2)` | 5.96e-8 | 1.86e-8 | 1.21e-7 | PASS |
| cosine Q1/Q2 | f32 | `(4,)` | 2.83e-7 | 1.53e-7 | 1.24e-6 | PASS |
| angle Q1/Q2 | f32 | `(4,)` | 1.53e-5 | 6.68e-6 | 3.26e-7 | PASS |
| selected target Q | f32 | `(4,2)` | 5.96e-8 | 2.98e-8 | 4.80e-7 | PASS |
| TD target | f32 | `(4,2)` | 5.96e-8 | 1.86e-8 | 2.33e-7 | PASS |
| critic total | f32 | `127.763290 / 127.763290` | 0 | 0 | 0 | PASS |
| actor total | f32 | `654.556458 / 654.556335` | 1.22e-4 | 1.22e-4 | 1.86e-7 | PASS |
| critic gradients | f32 | all layers | 2.29e-5 | 1.27e-7 | 1.55e-2* | PASS |
| actor gradients | f32 | all layers | 7.63e-5 | 5.53e-7 | 4.86e-2* | PASS |

`*` The largest relative gradient errors occur at reference entries near zero;
the recorded absolute errors are the primary criterion and remain below
`1e-4`. No parameter gradient tree is all zero.

## 9. Raw gradients, JIT, and VMAP

PyTorch uses `zero_grad(); loss.backward()` without optimizer steps. JAX uses
`jax.grad`. Every layer's weight and bias gradient is compared after applying
the same `[out,in] ↔ [in,out]` transpose. Adam state and clipped gradients are
outside this audit.

The JAX-only checks establish eager output equals `jax.jit` output and batched
Actor output equals `vmap` over its valid single-sample application. No
already-batched function is mechanically wrapped in another `vmap`.

## 10. Default-precision diagnostic retained

On an RTX 4090 with JAX 0.10.2, the default GPU float32 run produced 11 failed
and 16 passed tests. Float64 core tests, interpolator output, target critic
index, JIT, and VMAP passed. Representative default-GPU float32 mismatches are:

| Metric | PyTorch reference | JAX GPU | Max abs error | Frozen atol/rtol |
| --- | ---: | ---: | ---: | ---: |
| Actor output | `(4,6)` | `(4,6)` | 2.70e-4 | 2e-6 / 1e-5 |
| Critic Q1 | `(4,2)` | `(4,2)` | 1.38e-4 | 2e-6 / 1e-5 |
| Scalarized Q1 | `(4,)` | `(4,)` | 7.18e-5 | 2e-6 / 2e-6 |
| Cosine Q1 | `(4,)` | `(4,)` | 2.18e-4 | 2e-6 / 1e-5 |
| Angle Q1 | `(4,)` | `(4,)` | 1.68e-2 | 2e-4 / 2e-6 |
| Selected target Q | `(4,2)` | `(4,2)` | 1.20e-4 | 2e-6 / 1e-5 |
| TD target | `(4,2)` | `(4,2)` | 3.76e-5 | 2e-6 / 1e-5 |
| Critic raw gradient | all layers | all layers | 1.49e-2 | 1e-4 / 1e-4 |
| Actor raw gradient | all layers | all layers | 1.94e-2 | 1e-4 / 1e-4 |

Classification: **B. numerical precision difference**. Repeating the unchanged
core suite with `highest` on the same GPU passed, and the final suite including
the configuration-policy check gives `28 passed`. This isolates the discrepancy
to GPU float32 matrix-multiply precision rather than tensor layout or PD-MORL
formulas. The reproduction baseline therefore explicitly uses `highest`; the
backend-default failure remains recorded rather than being hidden. No tolerance
or algorithm formula was changed.

During audit construction, the exporter initially read critic gradients after
actor backpropagation and therefore accumulated an unrelated gradient;
snapshotting immediately after critic backpropagation fixed the reference
harness before tolerances were frozen.

This stage does not test environment physics, MuJoCo/Brax trajectory matching,
optimizer state, replay sampling, convergence, Pareto metrics, online refits,
or multi-GPU execution.
