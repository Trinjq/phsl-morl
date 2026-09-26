# PD-MORL Preference-Q Alignment Loss

## Scope

This step connects the fixed-shape pure-JAX interpolator from Step 5.4 to the
MO-TD3 critic and actor losses. It preserves the existing vector Bellman target,
whole-vector pessimistic target selection, HER, parallel update schedule, target
updates, and gradient clipping. Online interpolator evaluation and refitting are
outside this step.

The source audit uses the [official PD-MORL repository at commit
`35aa1bc`](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/commit/35aa1bccb31c149c3f3bb895c7c6ffa674e4b461)
and the [ICLR 2023 paper](https://openreview.net/pdf?id=aw-Fd8ne_F6).

## Paper formula and official behavior

The paper introduces a directional-angle term to align preferences and vector Q
values. Its notation is ambiguous about original `w` versus projected `wp`, and
its critic expression describes squared error. The executable PyTorch source is
more specific:

```text
wp = I(w)
g(wp, Q) = rad2deg(acos(clip(cosine_similarity(wp, Q), 0, 0.9999)))
```

The source adds the clamp and degree conversion that are not explicit in the
paper formula. It uses projected `wp` for every angle, Smooth-L1 regression for
the critics, and no coefficient on either critic angle mean. The implementation
therefore follows the source operation by operation.

## Roles of `w` and `wp`

Original replay preference `w` remains responsible for:

- actor and critic network conditioning;
- target-critic scalarization and whole-vector critic selection;
- actor objective `w^T Q1`.

The pure-JAX Step 5.4 forward computes `wp = interpolate(state, w)`. `wp` is
used only by `directional_angle`. It is not normalized, clipped, projected,
stored in replay, passed into the networks, or used in the TD target.

## Exact losses

For online critic vectors `Q1`, `Q2` and the unchanged vector Bellman target
`y`, the critic minimizes:

```text
mean(g(wp, Q1)) + smooth_l1_loss(Q1, y)
+ mean(g(wp, Q2)) + smooth_l1_loss(Q2, y)
```

PyTorch `F.smooth_l1_loss` uses its default `reduction="mean"`, so each critic's
loss is averaged over batch and objective dimensions before the two results are
added. The angle coefficient is implicitly one; `actor_loss_coeff` is never
applied to the critic.

The actor uses critic Q1 only and minimizes:

```text
-mean(w^T Q1) + actor_loss_coeff * mean(g(wp, Q1))
```

The official Walker setting defines `actor_loss_coeff=10`. EvoRL now reads the
same value from `configs/agent/mo-td3.yaml` and passes it into the agent instead
of hard-coding it inside the loss.

## Directional-angle boundaries

The existing `evorl/utils/morl_math.py` primitive was audited and already
matches the source. Its zero-gradient-safe norm remains unchanged.

| Relation between `wp` and Q | Cosine after clamp | Angle |
|---|---:|---:|
| Same direction | 0.9999 | approximately 0.810291 degrees |
| Orthogonal | 0 | 90 degrees |
| Opposite | 0 | 90 degrees |
| Positive rescaling | unchanged | unchanged |

Float32 evaluation can differ in the last few decimal places near the upper
clamp; tests use `1e-3` degrees there and exact source expectations elsewhere.

## State, JIT, and gradient paths

`PDMORLInterpolatorState` is stored under `AgentState.extra_state.interpolator`.
It contains only fixed-shape JAX arrays. Both losses call the pure-JAX
`interpolate`; no SciPy object, refit, host callback, or dynamic knot operation
enters JIT.

The critic optimizer differentiates both Smooth-L1 and angle paths with respect
to critic parameters. The actor optimizer differentiates through actor → Q1 for
both scalar-Q and angle terms while critic parameters remain fixed inputs. The
existing optimizer continues to apply one global-L2 norm clip of 100 to each
critic or actor update. Source clamp regions may produce zero angle gradients;
tests require finiteness rather than nonzero gradients.

The production workflow loads initial key solutions from the explicit
`interpolator_artifact` path and performs the Step 5.4 host-side SciPy fit before
training. The official repository does not ship `interp_objs_walker2d.txt`, so
no synthetic solution is placed in production configuration. Missing state
raises an explicit error. Tests alone use a clearly marked deterministic
fixture.

## Verification

Numerical tests cover same, orthogonal, opposite, positively rescaled, batched,
float32, eager, JIT, and `vmap` angle evaluation. Separate manual constructions
verify the exact critic and actor equations, including the unweighted critic
angles and configurable actor coefficient. Gradient tests cover normal,
upper-clamped, lower-clamped, and small-norm cases plus complete actor and critic
parameter PyTrees.

CPU regression results:

- Step 5.1 math + Step 5.4 Golden Test + Step 5.5 tests: 17 passed.
- Step 4.4 MO-TD3 and scalar-TD3 tests: 4 passed; the pre-existing real-Walker
  GPU-only test was deselected on this CPU host.
- Step 5.2 HER + Step 5.3 parallel exploration: 16 passed.
- Ruff checks passed for all changed Python files.

## Classification

| Item | Classification |
|---|---|
| `wp` angle input, clamp, degree conversion, two Smooth-L1 means, unweighted critic angles, Q1 actor loss, coefficient 10, delayed schedule | SOURCE-FAITHFUL |
| Fixed-array state in `AgentState`, explicit host artifact loading, JAX loss helpers, missing-state validation | FRAMEWORK-ADAPTATION |
| Synthetic deterministic solutions in tests only | TEST FIXTURE |
| Online key evaluation/replacement/refit, Pareto metrics, HV, sparsity, multi-GPU placement | NOT IMPLEMENTED |

There is no intentional deviation from the official loss behavior.
