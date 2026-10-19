> 历史记录（2026-10-10 统一标记）：本文保留当时配置、结果和阶段任务；其中“当前”“下一步”“必须”等仅适用于当时版本。现行算法、artifact、计数与运行入口以[当前基线协议](../../docs/PD_MORL_REPRODUCTION_PROTOCOL.md)为准。

# Key Artifact Refresh Validation

## Outcome

The refreshed three-key Walker2d Brax artifact was trained and compared against the previous artifact with two matched 2M-step runs per artifact. A post-promotion code audit found that its trainer averaged the two critic losses instead of summing them as the production workflow does, so the promotion has been withdrawn pending regeneration with the corrected trainer.

The withdrawn artifact is retained for provenance at:

configs/artifacts/interp_objs_walker2d_brax.txt

SHA-256:

cfdff6643b70591670080f1c80fa9a146c1d60576e1cca57813c1ed84ff843f2

The corrected refresh is now promoted as configs/artifacts/interp_objs_walker2d_brax_v3.txt (SHA-256 e8bce224c1b767d1588bde14a3560933a4f74262acd3982f2bd26889d79d3c93). The previous v2 artifact remains available as the historical active baseline.

## Provenance

- Repository: /home/qiuquanj/projects/phsl-jax-rbf-l2
- Branch: experiment/key-artifact-refresh-20261009
- Frozen source commit: 27b16052af1d91fca6b302583e2139a8424fd967
- Runtime: JAX 0.10.2, Brax 0.14.2, GPU backend, RTX 4090
- Candidate artifact: outputs/key_artifact_refresh/key_full/interp_objs_walker2d_brax.candidate.txt
- Candidate SHA-256: cfdff6643b70591670080f1c80fa9a146c1d60576e1cca57813c1ed84ff843f2
- Legacy SHA-256: 989e74d631ba74572884c4f32b19e12c9317af847ffbc6c02bdde5ad39bfa209

Candidate validation returned success for finite initial and online interpolation checks, artifact metadata, key ordering, endpoint/midpoint geometry, and JAX/SciPy interpolation agreement. The validator emitted CUDA preallocation and optional warp import warnings, but exited successfully.

## Withdrawn candidate key solutions

Keys are ordered as [0, 1], [0.5, 0.5], and [1, 0].

    [ 442.87188720703125, 2491.552978515625]
    [3090.75732421875, 2036.6268310546875]
    [3629.1171875,  467.25341796875]

## Old/new key-direction comparison

The angle between each legacy and refreshed L2-normalized key return vector is 0.70 degrees for [0, 1], 1.64 degrees for [0.5, 0.5], and 6.06 degrees for [1, 0]. The midpoint scalarized return changed from 2,466.85 to 2,563.69. The refreshed endpoint ordering remains consistent with the preference weights, and the midpoint is not directly dominated by both endpoints under the validator checks.

## Matched training results

The training command requested total_timesteps=2,000,960. The authoritative counters_summary.json for every run reports 780 host chunks, 3,120 inner rollouts, and exactly 2,000,960 environment transitions. The periodic train.log record at iteration 3,100 is intermediate; the final counter summary and training-final evaluation are used below.

| Artifact | Seed | Final hypervolume | Pareto points | Final sparsity | Mid preference return (R1, R2) | Key eval/refit/repl. | Env. steps |
|---|---:|---:|---:|---:|---|---|---:|
| Legacy | 0 | 6,854,922.53 | 199 | 2,224.23 | (2,871.46, 1,827.99) | 69 / 69 / 0 | 2,000,960 |
| Legacy | 1 | 6,030,473.57 | 122 | 3,607.16 | (1,948.00, 1,439.64) | 71 / 71 / 0 | 2,000,960 |
| Refreshed | 0 | 7,189,660.59 | 240 | 2,709.13 | (2,716.76, 1,880.27) | 67 / 67 / 0 | 2,000,960 |
| Refreshed | 1 | 6,089,595.55 | 113 | 690.77 | (1,066.87, 840.02) | 71 / 71 / 0 | 2,000,960 |

Across the two seeds, mean hypervolume changed from 6,442,698.05 to 6,639,628.07 (+3.06%), and mean Pareto count changed from 160.5 to 176.5. Mean sparsity changed from 2,915.69 to 1,699.95. The improvement is not uniform across every seed and metric: refreshed seed 0 improved hypervolume and Pareto count but had higher sparsity, while refreshed seed 1 improved hypervolume and sparsity but had fewer Pareto points. Because the candidate was trained with the wrong critic-loss reduction, these runs are historical observations and do not support promotion.

End-to-final-evaluation elapsed times were 2,528.3 s and 2,548.5 s for legacy seeds 0/1, and 2,550.8 s and 2,508.6 s for refreshed seeds 0/1. The first logged iteration-12 times, used as a compile/warm-up proxy, were 241.8 s, 241.7 s, 240.7 s, and 237.0 s in the same order. These are run-level observations on the lab4090 GPU, not a controlled speedup claim.

## Run outputs

- Legacy seed 0: outputs/key_artifact_refresh/old_seed_0
- Legacy seed 1: outputs/key_artifact_refresh/old_seed_1
- Refreshed seed 0: outputs/key_artifact_refresh/new_seed_0
- Refreshed seed 1: outputs/key_artifact_refresh/new_seed_1
- Candidate validation: outputs/key_artifact_refresh/evaluation/validation.json
- Promotion record: outputs/key_artifact_refresh/evaluation/promote.json
- Per-run counters: outputs/key_artifact_refresh/{old_seed_0,old_seed_1,new_seed_0,new_seed_1}/counters_summary.json

All four run metadata files record the frozen source commit, GPU backend, absolute artifact path, and the SHA corresponding to the artifact used by that run.

## Execution note

An initial refreshed seed-0 launch used a relative artifact path and resolved to the legacy SHA outside the repository. It was stopped before being counted, moved to outputs/key_artifact_refresh/new_seed_0_invalid_relative_old_sha, and excluded from all results. The valid refreshed runs used the repository artifact's absolute path and were independently checked from their logs and metadata.

## Verification

- Targeted tests: 57 passed, 1 warning
- Smoke run completed on the GPU path
- Candidate validator: successful
- Legacy and refreshed seed 0/1 runs: completed normally
- Withdrawn artifact SHA: verified as cfdff6643b70591670080f1c80fa9a146c1d60576e1cca57813c1ed84ff843f2
- Default artifact restored to the `_v2` artifact pending corrected retraining

## Corrected v3 promotion

The corrected trainer was run on the remote lab4090 GPU 0 in
outputs/key_fix_full. All three keys completed with:

- 2,000,128 actual environment steps and 25,088 random-action steps each;
- 1,999,929 Critic and 999,964 Actor optimizer steps each;
- sum_of_per_critic_mean_smooth_l1 loss aggregation;
- finite final losses and nonzero Actor loss;
- GPU JAX execution and successful JAX/SciPy interpolation validation.

Candidate SHA-256:
e8bce224c1b767d1588bde14a3560933a4f74262acd3982f2bd26889d79d3c93.

The v3 copy and metadata are versioned in configs/artifacts/; no historical
artifact was overwritten.
