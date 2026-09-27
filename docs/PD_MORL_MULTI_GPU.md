# PD-MORL Multi-GPU

## Scope and frozen baseline

Step8 maps the frozen PD-MORL single-GPU algorithm onto multiple local GPUs. It
does not change the production actor or twin critics (`[400, 400]`), logical
preference workers (`K=10`), global learner batch (`256`), HER, scalarization,
Bellman target, update schedule, evaluator, or Step6 tolerances.

## STEP8 MULTI-GPU AUDIT

1. The local Windows host has one NVIDIA GeForce RTX 4060 Laptop GPU (8188
   MiB), but its installed JAX exposes only a CPU device. The validation host
   `lab4090` has three NVIDIA GeForce RTX 4090 GPUs (24564 MiB each); JAX 0.10.2
   exposes all three CUDA devices.
2. EvoRL already contains `Mesh`, `NamedSharding`, `PartitionSpec`,
   `jax.shard_map`, `device_put`, collective helpers, and replicated gradient
   updates. It does not use `pmap` in the RL path.
3. The generic off-policy multi-device path is not source-faithful for PD-MORL:
   it divides `num_envs`, replay capacity, and warm-up counts by device count,
   and initializes one replay per device. PD-MORL must not use that path.
4. Actor, critics, targets, interpolator arrays, learner counters, and optimizer
   states are small enough to replicate on 24 GiB learner GPUs. Replication is
   also required so every device applies the same all-reduced update.
5. Rollout state is sharded over a leading physical-slot axis. Two GPUs use
   `5+5` slots. Three GPUs use `4+4+4`, with logical IDs `0..9` followed by two
   invalid slots.
6. Replay remains one controller-owned logical buffer. Valid rollout rows are
   gathered in logical-worker order before the one replay/HER insertion.
7. Replay sampling occurs once at size 256. Two GPUs receive 128 rows each.
   Three GPUs receive 258 physical rows (`86` each), with two repeated dummy
   rows masked invalid.
8. Each shard produces an additive loss/gradient contribution divided by the
   global valid denominator. `psum` forms the global gradient before the
   existing Optax chain applies global-norm clipping and Adam.
9. Orbax saves one global pytree. Replicated/sharded arrays restore through the
   supplied abstract state; the smoke test exercises one logical checkpoint,
   not per-device checkpoints.
10. Two GPUs need no padding. Three GPUs need two rollout slots and two learner
    rows of padding.
11. Main algorithm risks are accidental per-device replay, local clipping,
    equal-weight averaging of the `86/86/84` shards, device-derived RNGs,
    duplicate controller execution, and unmasked padding. The Step8 helpers
    and tests directly guard these cases.

## Architecture

Logical worker identity is independent of device placement. Worker RNG keys are
derived by folding in logical worker ID. Physical device ordinal is not an
algorithm input. The controller owns replay sampling, key evaluation,
interpolator refitting, recording, evaluation, and checkpointing. Learner state
is replicated; rollout and sampled learner rows are sharded.

The two-GPU layout is workers `0..4` on GPU0 and `5..9` on GPU1, with 128
learner rows per device. The three-GPU layout is workers `0..3`, `4..7`, and
`8,9,PAD,PAD`; learner shards contain `86,86,84` valid rows. Padding repeats a
valid row at the sharding boundary to avoid exceptional zero-vector angle
paths, then contributes exactly zero through the validity mask.

Global replay preserves the original insertion-order, capacity, size counter,
sampling population, and HER activation threshold. Target-policy noise is
generated once for the global sampled batch before learner sharding. Masked
Smooth-L1 uses denominator `256 * num_objectives`; masked angle and scalar terms
use denominator `256`.

The host runs the frozen `PDMORLEvaluator` and SciPy interpolator fit once. The
resulting fixed-shape arrays are replicated and checked shard-for-shard.

## Verification and current status

The following real-GPU checks passed on `lab4090`:

- two-GPU balanced layout, global batch and all-reduced toy learner update;
- optimizer clipping after global aggregation;
- replicated parameter equality;
- one global replay and checkpoint round trip;
- three-GPU `12/10` rollout layout and `258/256` learner layout;
- masked global reductions and exact padding-zero-effect;
- logical-worker PRNG placement invariance.

The production-shape real-Walker two-GPU smoke reached multi-device rollout,
one global replay/HER insertion, one global batch sample, and the distributed
critic comparison. The subsequent Step8.0 diagnosis below attributes the
gradient difference to float32 full-vs-split reduction order. The Step6
float32 gradient tolerance remains frozen at `atol=1e-4, rtol=1e-4`; no
tolerance or algorithm setting has been changed. Three-GPU production smoke
and final single-GPU regression remain paused by the requested stop condition.

## Classification

SOURCE-FAITHFUL: `K=10`, preference subspaces, one global replay, effective
batch 256, HER, learner losses, update schedule, clipping threshold/order,
control triggers, and evaluator semantics.

FRAMEWORK-ADAPTATION: logical-to-physical placement, replicated state,
gradient all-reduce, padded physical slots/rows, validity masks, host control,
interpolator broadcast, and sharded checkpoint placement.

TEST OVERRIDE: smoke episode length, warm-up/learner trigger, evaluation grid,
repeats, and duration only.

The requested Step8.0 stopping condition is now met. Distributed replay
optimization, multi-node work, PSL-MORL, and hypernetworks are out of scope.

## Distributed Numerical Parity Diagnosis

The frozen diagnostic uses production Walker dimensions (`obs=[256,17]`,
`action=[256,6]`, `preference=[256,2]`), `[400,400]` twin critics, float32
`matmul_precision=highest`, fixed critic parameters, fixed target values,
preferences, projected preferences, and target-noise seed `3`. The diagnostic
batch is generated once with seed `20260927`; all three references consume the
same arrays in the same order. The raw JSON evidence is
[`PD_MORL_NUMERICAL_DIAGNOSIS.json`](PD_MORL_NUMERICAL_DIAGNOSIS.json).

Reference A is the direct full-batch frozen loss. Reference B evaluates the two
`128`-row additive local contributions on one GPU and adds their raw gradients.
Reference C uses the real two-GPU `shard_map` path and one gradient `psum`
aggregation stage. Both distributed denominators are explicit: Smooth-L1
`256*2=512`, angle `256`.

| dtype | comparison | max abs | max relative L2 | minimum cosine |
|---|---|---:|---:|---:|
| float32 | full vs split | 6.8317e-2 | 1.7726e-3 | 0.99999870 |
| float32 | split vs 2-GPU | 0 | 0 | 1.00000000 |
| float32 | full vs 2-GPU | 6.8317e-2 | 1.7726e-3 | 0.99999870 |
| float64 | full vs split | 3.8725e-12 | 5.4822e-14 | 1.00000000 |
| float64 | split vs 2-GPU | 0 | 0 | 1.00000000 |
| float64 | full vs 2-GPU | 3.8725e-12 | 5.4822e-14 | 1.00000000 |

For float32, the largest element is `critic_0/hidden_2/bias[0]`:
`G_full=-56.8797340`, `G_split=G_2gpu=-56.8114166`, absolute difference
`6.83174e-2`, relative difference `1.20109e-3`. Every Q1/Q2 layer shows the
same pattern: full-vs-split differs, while split-vs-2-GPU is exactly zero in
the captured arrays. The raw per-layer table is retained in the JSON evidence;
the largest float32 full-vs-split relative-L2 error is Q1 layer-1 weight
(`1.7726e-3`, RMS `2.12396e-3`, cosine `0.99999870`).

Loss components show the same separation. Float32 full-vs-split total-loss
error is `4.42505e-4` (relative `2.66546e-6`); split-vs-2-GPU is exactly zero.
Float64 full-vs-split loss and split-vs-2-GPU loss errors are zero at reported
precision. The global denominator audit therefore passes; this is not a local
mean, double division, or padding error.

The raw gradient norms are float32 `450.5471` (full) versus `450.3713`
(split/2-GPU), and float64 `627.9073407` for all three. After global norm
clipping, float32 full-vs-split still differs (maximum `1.0241e-2`), while
split-vs-2-GPU remains zero. With the same Adam state, float32 full-vs-split
updated-parameter maximum error is `5.99979e-4`; split-vs-2-GPU remains zero.
The split-order test is exactly unchanged for the captured two-shard order.

### Classification

`G_split == G_2gpu`, the loss components agree on the same split path, and the
full-vs-split discrepancy shrinks from `6.8317e-2` in float32 to `3.8725e-12`
in float64. This is **STEP8 NUMERICAL REDUCTION-ORDER DIAGNOSIS**: a
float32 full-batch versus split-reduction effect, not a distributed
implementation bug. No tolerance, PD-MORL loss, batch size, network,
optimizer, clipping rule, or production setting was changed. The 3-GPU
production smoke remains paused as requested.
