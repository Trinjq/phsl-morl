# Archived Step8 data-parallel investigation

This document records the experimental multi-GPU PD-MORL data-parallel work.
The implementation is preserved on branch
`archive/pd-morl-step8-data-parallel` and is intentionally not imported by the
production single-GPU training path. The supported experiment architecture is
one complete K=10 PD-MORL training run per GPU, with independent training seed,
replay, optimizer, PRNG, interpolator, and checkpoint.

Do not use this archived document as a production launch recipe.
