# Motion datasets

All motion payloads live under this directory. Manifests remain outside it because
they describe lifecycle state (`current` or `test`) rather than carrying
motion data.

| Directory | Contents | Formal use |
|---|---|---|
| `stage1_full/` | LAFAN1, SEED-simple, MotionDecode skills, and other full Stage-I sequences | Yes |
| `motiondecode_backflip_grounded/` | 18 converted MotionDecode back-somersault takes | Selected subset |
| `seed_backflip_grounded/` | 28 converted SEED backflip takes | Selected subset |
| `seed_stunts_grounded/` | 232 converted SEED stunt takes, including cartwheels | Selected subset |
| `raw/` | Source CSV/BONES-SEED inputs used for conversion | Data preparation only |

`grounded` means the NPZ files were raised to satisfy the active G1 collision-geometry
clearance contract. It is part of the data transformation, not a report directory.

The lifecycle manifests reference these directories directly. A recorded
`data-datasets-v1` provenance migration changed path strings and regenerated strict
artifact hashes without changing any NPZ payload bytes.
