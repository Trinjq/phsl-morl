# PD-MORL Multi-Dimensional Preference Interpolator

## Scope and result

This subsystem reproduces the official host-side SciPy fit and exposes only a
fixed-shape, pure-JAX `I(w)` forward. It does not modify MO-TD3 losses, targets,
replay, evaluation, or the training loop. The worthwhile contribution is a
Golden-Tested SciPy-to-JAX boundary that preserves the official initial/update
normalization difference.

The source audit uses the official repository at commit
[`35aa1bc`](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/commit/35aa1bccb31c149c3f3bb895c7c6ffa674e4b461)
and the [ICLR 2023 paper](https://openreview.net/pdf?id=aw-Fd8ne_F6).

## Paper semantics

Appendix B.1.3 defines key preferences as every one-hot vector plus the uniform
vector `[1/L, ..., 1/L]`. Key solutions approximate Pareto solutions at those
preferences, are normalized to unit vectors, and fit a multidimensional
interpolator `I(w)`. Its output `wp` aligns preferences with objective values;
the interpolator may be updated when training finds better key solutions.

The terms used here are distinct:

- `w`: an original preference queried by the policy.
- key preference: an anchor input to the interpolator.
- key solution: the objective-return vector obtained at an anchor.
- normalized key solution: the fitted target vector.
- `wp = I(w)`: an objective-space direction, not necessarily a probability
  vector and not required to sum to one.

## Official source behavior

### Key preferences and their order

The key-training script builds a simplex grid with `itertools.product`, filters
rows whose sum is one, applies `np.unique`, and selects three equally spaced
indices. Consequently, Walker's actual order is:

```text
[0.0, 1.0]
[0.5, 0.5]
[1.0, 0.0]
```

This differs from the paper's illustrative ordering. `key_preferences(2)`
preserves the source order. For general `L`, it returns the paper's one-hot plus
uniform set in NumPy lexicographic order; only the Walker `L=2` ordering is
claimed as source-verified.

### Key-solution source

[`train_Walker2d_MO_TD3_HER_Key.py`](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/blob/35aa1bccb31c149c3f3bb895c7c6ffa674e4b461/PD-MORL/train_Walker2d_MO_TD3_HER_Key.py)
starts one MO-TD3 process per key anchor. It uses the ordinary
`ExperienceReplayBuffer`, not the HER buffer. Evaluation uses the original
fixed key and stores the objective return from the best scalarized evaluation.
Training is centered on the fixed key, but the official agent adds clipped
Gaussian preference noise and L1-normalizes it at every non-deterministic
action; the perturbed preference is stored with the transition. Thus “fixed
preference” describes the anchor and evaluation objective, not literally
constant training inputs.

The resulting rows are written to `interp_objs_walker2d.txt`, which the main
script loads. That artifact is not committed in the official repository and is
not present in EvoRL. Tests therefore use clearly marked deterministic synthetic
solutions. They are not official Walker results or paper data.

### Fit configuration and normalization

[`train_Walker2d_MO_TD3_HER.py`](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/blob/35aa1bccb31c149c3f3bb895c7c6ffa674e4b461/PD-MORL/train_Walker2d_MO_TD3_HER.py)
constructs exactly:

```python
RBFInterpolator(key_preferences, normalized_key_solutions, kernel="linear")
```

No other argument is explicit. In the tested SciPy 1.17.1 API this means
`neighbors=None` (all knots), `smoothing=0`, `epsilon=1` for the linear kernel,
and the linear kernel's minimum/default `degree=0`. The kernel is `-r` and the
degree-zero polynomial contributes one constant basis column.

The source intentionally has two different target transformations:

- Initial fit: `x / ||x||_2` row by row.
- Every online refit: `x / ||x||_1` row by row.

They are reproduced rather than unified. Zero-row protection divides by one;
this is a framework adaptation that leaves every normal nonzero row unchanged.

### Online update rule and timing

The main process evaluates all key preferences when every worker's episode
counter is strictly greater than `eval_cnt_ep`, then increments that threshold
by one. With ten workers this is a synchronized per-worker episode boundary,
not the separate `eval_freq=100` full Pareto evaluation schedule. Each candidate
is averaged over `eval_episodes=3` evaluations and replaces its old row iff:

```text
key_preference @ candidate_return > key_preference @ old_return
```

There is no tolerance, averaging with the old row, or non-dominance test. After
all comparisons, the source L1-normalizes every row and creates a new
`RBFInterpolator`, even if no row changed.

### How MO-TD3 uses `wp`

[`lib/common_ptan/agent.py`](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm/blob/35aa1bccb31c149c3f3bb895c7c6ffa674e4b461/lib/common_ptan/agent.py)
calls `I(w)` for critic and actor batches and passes the result directly to
cosine-similarity/directional-angle terms. It does not L1/L2-normalize, clip,
softmax, or project `wp` afterward. Original `w`, not `wp`, scalarizes Q values.

## Implementation boundary

`fit_reference_interpolator` is the offline reference and refit path. It calls
SciPy on the host. `fit_interpolator_state` extracts the knots, normalized
targets, RBF/polynomial coefficients, polynomial shift/scale/powers, and epsilon
into `PDMORLInterpolatorState`; the SciPy Python object is discarded.

`interpolate(state, w)` reconstructs SciPy's evaluation matrix in JAX:

```text
[-epsilon * ||w - key_i||_2 for each key_i, polynomial terms] @ coefficients
```

It accepts `[L]` and `[..., L]`, has fixed shapes, contains no NumPy, SciPy,
callback, or Python interpolator at runtime, and works with eager execution,
`jax.jit`, native batching, and `jax.vmap`. Host fit/refit is deliberately not
JIT compiled. A future training integration should replace state arrays only at
an evaluation/control boundary without changing their shapes.

`update_key_solution` implements only the source comparison. Calling code must
then invoke `fit_interpolator_state(..., normalization="online")`; this step
does not connect either operation to online training.

## Golden test and numerical error

`tests/test_pd_morl_interpolator.py` checks both initial-L2 and online-L1 fits at
all three knots, interior points, a midpoint, near-boundary simplex points, and
the complete batch. It covers single input, batch input, eager, JIT, `vmap`,
float32, float64, the update decision, and online refit. Tolerances are
`2e-7` absolute for float32 and `2e-14` for float64.

Against a SciPy float64 reference, the deterministic synthetic fixture produced:

| Path | JAX dtype | Max abs. error | Mean abs. error | Max rel. error |
|---|---:|---:|---:|---:|
| Initial L2 | float32 | 5.5143e-08 | 2.1924e-08 | 2.5684e-07 |
| Initial L2 | float64 | 5.5511e-17 | 4.6259e-18 | 2.3280e-16 |
| Online L1 | float32 | 3.7676e-08 | 1.4950e-08 | 1.9573e-07 |
| Online L1 | float64 | 0 | 0 | 0 |

The float32 error is the expected coefficient/input cast from the SciPy
float64 fit. No loose relative tolerance is used near small output components;
the Golden Test uses the measured absolute error envelope.

## Classification

| Item | Classification |
|---|---|
| Walker key order; linear RBF arguments; initial L2; online L1; strict scalarized replacement; raw `wp` output | SOURCE-FAITHFUL |
| Host-side SciPy coefficient extraction; fixed-shape JAX state/forward; zero-row safety | FRAMEWORK-ADAPTATION |
| Synthetic deterministic key solutions used only by tests | TEST FIXTURE |
| Unified normalization, approximate JAX fit, post-processing `wp`, or training integration | DEVIATION — not implemented |

No actor loss, critic loss, TD target, replay, HER, directional-angle primitive,
policy frequency, evaluation loop, or workflow state was changed in this step.
