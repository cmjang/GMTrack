# Ex-GRMT

Reproduction of **Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust
Generalist Humanoid Control**, pinned to the bundled `2607.20110v1.pdf` (SHA256
`55cca5c02f16c659e4ab3baf08d9ad1fb69865f37cfba084958ade0911cf51fe`), on
the Unitree G1, built as an external task package for
[mjlab](https://github.com/mujocolab/mjlab).

The method is a two-stage continual-learning framework:

- **Stage I** trains a generalist motion-tracking base policy over a multi-source
  motion distribution, using a history-conditioned command encoder with a finite
  scalar quantization bottleneck.
- **Stage II** expands that policy toward highly dynamic skills (backflips, kip-ups,
  aerial cartwheels) with **PACE** — asymmetric acquisition/consolidation environment
  roles plus a progress-adaptive regularizer toward the frozen base policy — and
  **STAR**, which resamples high-advantage trajectory fragments from difficult
  temporal regions.

## Setup

```bash
uv sync
uv run list-envs | grep ExGRMT
```

`third_party/rsl_rl` is a local editable checkout of `rsl-rl-lib` v5.4.0 (the version
mjlab pins). It is not patched — all modifications live in `src/ex_grmt/rsl_rl/` and
plug in through rsl-rl's `class_name` mechanism.

## Quick start

```bash
# 1. Build complete Stage-I sequences (no 10-second slicing here)
uv run python -m ex_grmt.scripts.prepare_motions \
    --input-dir data/datasets/raw/lafan1 --source lafan1 --input-format mjlab --input-fps 30 \
    --output-dir data/datasets/stage1_full/lafan1 \
    --manifest logs/data_build/manifests/lafan1_full.json
uv run python -m ex_grmt.scripts.prepare_motions \
    --input-dir data/datasets/raw/seed_simple --source seed-simple --input-format mjlab --input-fps 50 \
    --output-dir data/datasets/stage1_full/seed_simple \
    --manifest logs/data_build/manifests/seed_simple_full.json
uv run python -m ex_grmt.scripts.curate_seed_simple merge \
    --base-manifest logs/data_build/manifests/lafan1_full.json \
    --additional-manifest logs/data_build/manifests/seed_simple_full.json \
    --output-manifest logs/data_build/manifests/stage1_lafan_seed_simple.json
# Build the strict duration-matched public-data proxy. Select it explicitly with
# EX_GRMT_STAGE1_MANIFEST when running the paper-aligned source mix.
uv run python scripts/build_stage1_paper_manifest.py

# 2. Stage I
uv run train ExGRMT-Stage1-Flat-Unitree-G1

# 3. Only now split >10 s sequences and run exactly 5 rollouts at the 80% threshold.
uv run python -m ex_grmt.scripts.stratify --checkpoint logs/rsl_rl/ex_grmt_stage1/<run>/model_99999.pt

# 4. Stage II
uv run train ExGRMT-Stage2-Flat-Unitree-G1 \
    --agent.algorithm.base-checkpoint logs/rsl_rl/ex_grmt_stage1/<run>/model_99999.pt

# 5. Evaluate one independently trained checkpoint (repeat for seeds 42...46)
uv run python -m ex_grmt.scripts.evaluate --checkpoint <ckpt> --training-seed 42 \
    --manifest data/current/stage1-116500-final30-cartwheel-balanced-nofall-probe/challenging.json \
    --out logs/eval/seed42.json
uv run python -m ex_grmt.scripts.aggregate_evaluations \
    --inputs logs/eval/seed42.json logs/eval/seed43.json logs/eval/seed44.json \
             logs/eval/seed45.json logs/eval/seed46.json \
    --out logs/eval/five_seed_summary.json
```

For SONIC-style closed-loop root orientation, use the separate Heading task family:

```bash
uv run train ExGRMT-Stage1-Heading-Flat-Unitree-G1
uv run train ExGRMT-Stage2-Heading-Flat-Unitree-G1 \
    --agent.algorithm.base-checkpoint logs/rsl_rl/ex_grmt_stage1_heading/<run>/model_*.pt
```

These tasks append the current-robot-to-reference pelvis orientation (6D) to every
command token, making heading error directly observable to the Actor. The original
task IDs remain 38D and checkpoint-compatible with existing runs; Heading is 44D and
must use its own Stage-I checkpoint. At deployment activation, align the reference's
initial yaw to the measured robot yaw once, then keep that offset fixed and recompute
the relative orientation from the live robot state every policy step.

The registered Stage-I task defaults to the formal 312-clip source under
`data/current/`. Rebuild intermediates are generated under `logs/data_build/` and
are not retained as runtime data; select an alternative explicitly through
`EX_GRMT_STAGE1_MANIFEST`. The formal source's backflip component is the
user-screened final set: 14 MotionDecode + 16 SEED takes, plus the selected SEED
stunt slice and 20 representative cartwheel clips. All
substitutes are aligned upward against all active Instinct G1 collision geoms and
independently gated at 3 mm; this cleanup is a local proxy choice, not a
2607.20110v1 parameter. Old-run configuration snapshots remain under `logs/`, but
retired local manifests and one-off previews are not retained in `data/`.
Stage II accepts only provenance-bearing
`stratified.json`, `mastered.json` and `challenging.json` generated from the exact
Stage-I checkpoint being used. Hand-written source/category splits fail closed.
The paper's Table IV uses
3.096 h total: LAFAN1 (2.444 h), AMASS (0.511 h),
and in-house Xsens captures (0.141 h), all retargeted to G1 and resampled to
50 Hz. The immediately usable public source here is
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset);
place its CSVs under `data/datasets/raw/lafan1/`. BONES-SEED/SONIC is not a paper training
source. The current proxy uses the small MotionDecode/SEED high-dynamic slice
described above; neither full corpus matches the paper's data distribution.

The active local motion layout is consolidated by purpose:
`data/datasets/stage1_full/` contains the general Stage-I sequences, while the final
30 screened backflips live in the two collision-grounded source directories and are
referenced directly by the formal manifest. Active and held-out artifacts are
separated by lifecycle; current
manifests contain no dangling motion paths.

Grounded conversions use `prepare_motions --ground-alignment g1_collision` and
store finite clearance/correction/velocity QC in every schema-v4 NPZ. Before a
manifest enters training, validate it with
`python -m ex_grmt.scripts.audit_ground_clearance --threshold 0.003`; the audit
fails closed on any collision-geometry violation or inconsistent metadata.
Clearance and curation reports are reproducible QC output and are not retained in
the formal data layout. See [`data/README.md`](data/README.md) for the active,
held-out and rebuild layout.

If the complete private/retargeted datasets already exist elsewhere, point the task
registry at them without editing source code:

```bash
export EX_GRMT_STAGE1_MANIFEST=/abs/path/to/stage1_complete_sequences.json
export EX_GRMT_STRATIFIED_MANIFEST=/abs/path/to/stratified.json
export EX_GRMT_MASTERED_MANIFEST=/abs/path/to/mastered.json
export EX_GRMT_CHALLENGING_MANIFEST=/abs/path/to/challenging.json
```

The repositories under `cankao/` are used to resolve unpublished implementation
details. Most are low-level G1 choices (action scaling, actuator parameters,
collision geometry and push ranges); for the FSQ level count omitted by Ex-GRMT, the
SONIC training release's matching 2 x 32 tokenizer supplies the explicit 32-level
proxy. They contain no RGMT/Extreme-RGMT source, so the two papers remain
authoritative for PACE, STAR, the encoder and the paper-specific randomization
ranges, and the SONIC-derived level count remains an assumption rather than a paper
value.

The strict v1-proxy task still keeps `recovery_probability = 0`. The dedicated
recovery implementation is documented in [`docs/recovery_proxy.md`](docs/recovery_proxy.md),
follows RGMT Sec. II-D, and does not use get-up demonstrations or AMP/MJLab data.
It resets 15% of environments into randomized unstable poses, keeps the ordinary
motion reference advancing, anneals an upward `U[0,200] N` assist to negligible,
and applies a 3 s instability shield. The exact pose sampler and the 2.4M-step
anneal horizon are local assumptions listed there.

See [CLAUDE.md](CLAUDE.md) for the paper→code map, the full constant tables, the list
of assumptions where the paper is underspecified, and the known gotchas.

## Status

The implementation is locally validated, but no paper-scale reproduction result is
claimed yet. Paper-scale numbers require
cluster training (4096 envs × 100k iterations ≈ 9.8B environment steps for Stage I
alone, with 1024 environments on each of four GPUs); see the maintained scripts
under `scripts/slurm/`. Checkpoints produced before the current
paper-alignment fixes must not be used for final results: retrain Stage I, rerun
stratification, and then retrain Stage II.
