# PD-MORL Control and Evaluation Plane

## Scope

Step 6.1 adds only host-side control and evaluation behavior. The frozen actor,
critic, scalarization, Bellman target, interpolator forward, alignment losses,
gradient path, and Step 6 tolerances are unchanged.

## Official source audit

The audit targets upstream commit
`35aa1bccb31c149c3f3bb895c7c6ffa674e4b461` and the following files:

- `PD-MORL/train_Walker2d_MO_TD3_HER.py`
- `PD-MORL/train_Walker2d_MO_TD3_HER_Key.py`
- `lib/utilities/MORL_utils.py`
- `lib/utilities/settings.py`
- `PD-MORL/eval_benchmarks_MO_TD3_HER.py`

The source behavior is:

| Item | Executable source behavior |
|---|---|
| Initial solutions | Load `interp_objs_walker2d.txt`, produced by the three-process key-pretraining script |
| Shape/order | `[3,2]`, rows correspond to `[0,1]`, `[0.5,0.5]`, `[1,0]` |
| Initial fit | Row-wise L2 normalization, SciPy linear `RBFInterpolator` |
| Online fit | Row-wise L1 normalization, followed by a new linear `RBFInterpolator` |
| Counters | `eval_cnt = 1`, `eval_cnt_ep = 1` |
| Key trigger | `(process_episode_array > eval_cnt_ep).all()`; increment counter before evaluation |
| Full trigger | `(process_episode_array > args.eval_freq * eval_cnt).all()`; Walker `eval_freq=100`; increment before evaluation |
| Trigger ordering | Key update block precedes the full-evaluation block |
| Key repeats | `args.eval_episodes = 3` through the actual `eval_agent_interp` call |
| Candidate | Per-objective mean over repeat returns |
| Seeds | `eval_ep * 11`; training/key `[0,11,22]`; offline default `[0,11,22,33,44,55]` |
| Reset order | For each repeat, seed once, then reset/evaluate preferences serially in grid order |
| Policy | `deterministic=True`; preference reset occurs after each preference episode |
| Return | Undiscounted vector reward sum until done/episode limit |
| Training grid | Step `0.005`, 201 preferences; three repeats |
| Final/offline grid | Step `0.001`, 1001 preferences; training final three repeats, offline six repeats |
| Training aggregation | Compute HV/sparsity per repeat, then mean; objective returns mean across repeats |
| Offline aggregation | Retain per-repeat HV/sparsity/objectives, report mean/std, then filter the mean objectives |
| Pareto | `NonDominatedSorting().do(-returns, only_non_dominated_front=True)`; duplicate non-dominated rows remain |
| Hypervolume | `pymoo`, zero reference, evaluated on negative returns for minimization convention |
| Sparsity | For each repeat, Pareto-filter its returns, then sum squared adjacent sorted gaps over objectives and divide by `N-1`; `N<=1` returns zero |

## Initial artifact and provenance

Production config explicitly loads
`configs/artifacts/interp_objs_walker2d.txt`. There is no zero, random, synthetic,
or temporary production fallback. A missing file, non-finite value, wrong shape,
or mismatched `.npz` key ordering raises an error.

The vendored artifact contains the upstream Walker values:

```text
497.2413299560547,2494.5884033203124
1639.5962036132812,2156.5128173828125
2602.103564453125,691.5010070800781
```

- path: `configs/artifacts/interp_objs_walker2d.txt`
- SHA256: `7b7f701574ced75af9df1fefdece160463643a85df9826544f7fb2449257886b`
- shape: `[3,2]`
- parsed dtype: NumPy `float64`
- upstream source-file SHA256 (CRLF/trailing-space form):
  `7076c686661a4a560cb0e57fa9e3f4aed43fb30b546df81788d95e09a6c21816`

These are artifact values for this traced Walker experiment, not universal
Walker constants. Provenance, key ordering, dtype, shape, and raw values are
loaded and retained by the workflow.

## Worker counters and key updates

`episode_count[K]` is fixed-shape JAX state and accumulates terminal episodes
per logical lane, including replay prefill. With the strict Walker baseline,
`B=K=10` and `rollout_length=1`, each host control check follows one environment
step. Worker step counters and episode counters never reset and are not
interchanged with learner updates, global timesteps, or replay size.

The off-by-one rule is strict `>`: with the source initial counter `1`, all
workers at `2` trigger, while any worker at `1` prevents the trigger. The
counter increments before candidate evaluation. Each candidate is the vector
mean over three deterministic episodes. Replacement uses strict scalar score
improvement and direct assignment—no tolerance, EMA, Pareto condition, or
old/new averaging. The online L1 refit runs after every trigger even when no
candidate wins, and only the values of the fixed-shape JAX interpolator state
change.

## Evaluation and metrics

`PDMORLEvaluator` preserves repeat-then-preference logical ordering. Brax maps
the official integer repeat seed to `PRNGKey(seed)` and folds in the serial
preference index so successive resets are distinct. This mapping is a
FRAMEWORK-ADAPTATION; the logical seeds and aggregation order remain source
faithful.

`MORLEvaluationResult` contains preferences, `[S,M,L]` repeat returns, mean
returns, per-repeat HV/sparsity, their means, and the non-dominated indices and
returns of the mean front. Training control uses the 201-point/three-repeat
path. `evaluate_offline` uses the separate 1001-point/six-repeat path.

Pareto filtering is maximization-based and preserves duplicates. Hypervolume
uses the installed `pymoo` implementation with zero reference and the official
negative-return conversion. Sparsity now follows the source order independently
for every repeat: non-dominated filter, gap formula, then repeat mean. The
previous evaluator incorrectly applied the gap formula to all returns. Empty or
single-point fronts return zero.

## JIT boundary

The learner JIT contains only the existing fixed-shape pure-JAX interpolator
forward. Artifact I/O, deterministic key evaluation, candidate aggregation,
strict replacement, SciPy construction, L1 refit, Pareto filtering,
hypervolume, and sparsity run in `_after_multi_steps` or explicit offline
evaluation on the host. No SciPy object enters `AgentState` or `WorkflowState`.

## Classification

| Classification | Step 6.1 behavior |
|---|---|
| SOURCE-FAITHFUL | Artifact-derived solutions; key order; initial L2/online L1; counters and strict triggers; repeat mean; strict direct replacement; seeds; grids; undiscounted returns; aggregation; Pareto sign; zero-reference HV; sparsity |
| FRAMEWORK-ADAPTATION | Multiprocessing episode counters become fixed JAX logical-lane counters; SciPy refit runs at a host hook and exports arrays; Brax uses `PRNGKey(seed)` plus serial-position folding |
| TEST FIXTURE | Synthetic key solutions and synthetic episode callbacks occur only in isolated tests |
| DEVIATION | Zero/random fallback, changed timing, tolerance replacement, EMA, Pareto deduplication, changed reference point, unified normalization, or SciPy inside learner JIT are prohibited and absent |

## Verification

Unit tests cover required artifact failures, provenance/order, counter
boundaries, candidate mean and replacement cases, refit behavior and shape,
seed lists, deterministic/undiscounted aggregation, 201/1001 grids, training
and offline aggregation, Pareto edge cases and duplicates, pymoo HV, and source
sparsity boundaries. Regression includes Step 5.1–5.5, Step 6 Golden Tests,
MO-TD3, HER, parallel exploration, and scalar TD3 without changing frozen
tolerances.

Final laboratory verification used JAX 0.10.2 on an NVIDIA RTX 4090 with
`pytest --device gpu`. The frozen regression set completed with `123 passed`,
including the hard `jax.default_backend() == "gpu"` real-Walker assertion.
The additional real Brax Step 6.1 evaluator smoke produced finite vector
returns, hypervolume, and sparsity with shape `[3,3,2]` on `cuda:0`.

## Status

`STEP6.1 PASS`

## Remaining Step 7 work

Step 7 end-to-end refactoring, multi-GPU, PSL-MORL, hypernetworks, and the
one-million-step paper reproduction remain intentionally out of scope.
