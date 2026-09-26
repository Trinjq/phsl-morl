# PD-MORL preference relabeling / HER

## Paper semantics

For each transition `(s, a, r_vec, s', done, w)`, the paper retains the
original and describes `N_w` additional transitions that differ only in their
preference. State, action, vector reward, next state, and termination data are
unchanged. This is preference relabeling, not goal-state HER, and reward is not
recomputed or scalarized.

## Source behavior

The official implementation is `ExperienceReplayBuffer_HER_MO._add` in
`lib/common_ptan/experience.py:174-198`. Walker uses `weight_num=3`,
`start_timesteps=10000`, `process_count=10`, and `replay_size=2000000`
(`lib/utilities/settings.py:56-76`).

For every base transition the source samples independent standard normals,
takes their absolute values, L1-normalizes each vector, and rounds to three
decimals. It writes the original first. Only after that write, it tests:

```text
len(buffer) > start_timesteps * process_count
```

If true, it appends the three relabeled entries in sampled order. Thus the
Walker threshold is 100,000 replay entries and a post-warm-up base transition
produces four physical entries. There is no rejection loop, so the source does
not explicitly guarantee that a sampled preference differs from the original.

For a vectorized add with threshold 10, initial replay length 8, four base
transitions, and `N_w=3`, the JAX `lax.scan` preserves sequential source
semantics:

| Base | Length after original | HER active | Length after base |
| --- | ---: | --- | ---: |
| 0 | 9 | no | 9 |
| 1 | 10 | no | 10 |
| 2 | 11 | yes | 14 |
| 3 | 15 | yes | 18 |

The per-base activation mask is therefore `[false, false, true, true]`; it is
not a single batch-wide decision based on the size before or after the batch.

The source generates its random HER preferences before checking warm-up, so
inactive transitions still advance NumPy's global RNG. JAX likewise executes
the sampler for every base transition, including inactive ones. Because JAX
uses explicit stateless keys rather than a mutable NumPy RNG, exact seeded
cross-framework random sequences are not preserved; the enabled samples still
follow the source distribution independently.

At full capacity, the source removes `1 + N_w` entries from the front and then
appends the original followed by all relabels. One source edge case exists:
when a group begins just below capacity and crosses it, the Python list can
exceed nominal capacity by up to `N_w`. Default Walker arithmetic reaches
2,000,000 on a four-entry boundary, so this bug is not exercised there.

## JAX mapping

**SOURCE-FAITHFUL:** sampler distribution and rounding, strict `>` warm-up
test after the logical original insertion, original-then-relabel ordering,
physical add-time entries, only replacing
`extras.policy_extras.preference`, and one uniformly sampled logical replay
pool.

**FRAMEWORK-ADAPTATION:** a dtype-tiny denominator guard covers the
measure-zero all-zero normal draw. Fixed-shape `[B,1+N_w,...]` groups are
flattened in source order and passed once to EvoRL's JIT-compatible ring
buffer with a mask for inactive relabels. The ring keeps its configured fixed
capacity; for unaligned custom capacities it evicts immediately instead of
reproducing the source's temporary/continuing over-capacity list bug.

The implementation does not create a second HER buffer or any per-worker or
per-preference pool; it writes into the workflow's existing logical replay
pool. Changes to EvoRL's existing multi-GPU sharding architecture are outside
this step.

## Paper / source differences

- The paper describes relabeling each transition; source code delays it until
  the strict replay-length warm-up condition is satisfied.
- The paper says random preferences; source code specifically uses
  absolute-normal samples, L1 normalization, and three-decimal rounding.
- The paper describes additional preferences as different, but source code
  does not reject equality with the original preference.

## Memory / replay analysis

The current Walker replay entry stores float32 payloads: observation 17
(68 B), action 6 (24 B), vector reward 2 (8 B), preference 2 (8 B), original
next observation 17 (68 B), termination (4 B), and truncation (4 B): about
184 bytes per entry. At 2,000,000 entries this is approximately 368 MB
(351 MiB), excluding array/runtime overhead.

The configured capacity remains 2,000,000 entries. HER therefore does not
multiply physical preallocation. Before warm-up, one environment transition
adds one entry; afterward it adds up to four, filling and evicting about four
times faster and reducing the retained base-transition horizon to roughly one
quarter. Expanding capacity to 8,000,000 to preserve the old base-transition
horizon would be an additional design and is not done here.

## Verification

- HER unit tests on GPU: `6 passed`.
- HER plus Step 4.1-Step 5.1 regression on GPU: `33 passed, 1 deselected`.
- Complete MO-TD3 regression including Brax Walker on GPU: `5 passed`.
