# Current cartwheel training data

This directory is the active representative-cartwheel dataset version.

- `seed_cartwheel_balanced_grounded.json`: 20 training clips (10 original
  `cartwheel_R` takes plus their 10 mirrors), 101.04 seconds.
- `stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json`:
  312 complete source sequences, 2.8592 hours, with no `cartwheelin` or
  `fallAndGetUp` clips.
- `stage1-116500-final30-cartwheel-balanced-nofall-probe/`: strict five-rollout
  Stage-II split with 1084 logical clips, `D_m=907`, and `D_c=177`.
- Stage-II artifact set: `92c4c87b0b556b37db89239b7b3f778564e5e606b8457ba91000e8b2cc7dc313`.

Motion NPZ files live under `data/datasets/` and are immutable payloads shared by
current and test manifests. The authenticated JSON files now reference
`data/datasets/` directly.

The strict four-file bundles preserve their generation-time provenance strings and
shared artifact hashes. Some provenance strings mention the path used on the machine
that generated them; do not hand-edit those strings, because that would invalidate
the authenticated bundle.
