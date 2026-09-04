# GMTrack

Highly dynamic humanoid whole-body motion tracking on the Unitree G1 (29 DoF), packaged
as an external task package for [mjlab](https://github.com/mujocolab/mjlab).

[![Built on mjlab](https://img.shields.io/badge/built%20on-mjlab-4c1.svg)](https://github.com/mujocolab/mjlab)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)

GMTrack is a customized implementation based on SONIC, Extreme-RGMT, RGMT, and
related humanoid-control work, adapted to the mjlab/BeyondMimic stack.

mjlab already ships a BeyondMimic reimplementation under `mjlab/tasks/tracking/`, so
GMTrack plugs into mjlab instead of forking it: tasks register through mjlab's
entry-point hook, and the learning code subclasses rsl-rl.

## Method

```
D — all motion data
│
├─ Stage I ─ plain PPO + adaptive bin sampling over all of D ──────────────► π_base
│
├─ Stratification (runs only after Stage I)
│     sequences > 10 s are cut into 10 s logical segments, short ones kept whole;
│     π_base is rolled out 5× per segment under randomization
│     completion ≥ 80% → D_m (mastered)          otherwise → D_c (challenging)
│
└─ Stage II ─ ONE training run, two environment groups, one shared network
      acquisition   (ξ = 0.8): samples D_c, adaptive sampling, PPO + STAR,
                               domain randomization / pushes / observation noise
      consolidation (1 − ξ)  : samples D_m, uniform sampling, nominal dynamics,
                               distillation toward the frozen π_ref = π_base
                                                                          ► π_aug
```

- **PACE** — asymmetric acquisition/consolidation environment roles plus a
  progress-adaptive regularizer `λ_con` toward the frozen base policy, so learning new
  highly dynamic skills does not erase mastered ones.
- **STAR** — advantage-prioritized resampling of trajectory fragments drawn from the
  hardest temporal bins, which compensates for the sparse, failure-heavy samples that
  extreme motions produce.
- **Policy** — a three-branch encoder (proprioceptive state, past actions, command
  window) feeding a causal attention encoder with a finite-scalar-quantization
  bottleneck. The actor consumes only deployable observations.

## Status

Research code under active development.

- The implementation is complete and covered by unit tests, and one end-to-end chain
  (Stage I → stratification → Stage II) has run to completion.
- **No benchmark numbers are published yet** — the five-seed sweep over the four test
  sets has not been run.
- Task configs, manifests, and interfaces may still change between commits.

## Installation

Requirements:

- Linux x86_64 with an NVIDIA GPU — mjlab simulates through MuJoCo Warp, which has no
  CPU backend
- Python 3.10–3.13
- [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/cmjang/GMTrack.git
cd GMTrack
uv sync
uv run list-envs | grep GMTrack
```

`uv sync` resolves torch from the CUDA 12.8 index. `uv run` re-syncs the environment on
every invocation; call `.venv/bin/python -m ...` directly to skip that.

`third_party/rsl_rl` is a local editable checkout of `rsl-rl-lib` 5.4.0, the version
mjlab pins exactly. It is never patched — every modification lives in
`src/gmtrack/rsl_rl/` and is injected through rsl-rl's `"module:Class"` resolution.

## Pipeline

### 1. Prepare motions

`prepare_motions` writes one complete sequence per source CSV — no 10 s slicing happens
here.

```bash
uv run python -m gmtrack.scripts.prepare_motions \
    --input-dir data/datasets/raw/lafan1 --source lafan1 \
    --input-format mjlab --input-fps 30 \
    --output-dir data/datasets/stage1_full/lafan1 \
    --manifest logs/data_build/manifests/lafan1_full.json
```

High-dynamic proxy clips are additionally aligned against the G1 collision geometry and
must pass the clearance gate before entering a manifest:

```bash
uv run python -m gmtrack.scripts.prepare_motions \
    --input-dir data/datasets/raw/seed_stunts --source seed-stunts \
    --input-format bones-seed --input-fps 120 --output-fps 50 \
    --output-dir data/datasets/seed_stunts_grounded \
    --manifest logs/data_build/manifests/seed_stunts_grounded.json \
    --ground-alignment g1_collision --ground-clearance-m 0.003 \
    --ground-smoothing-radius-s 0.3 --device cuda:0

uv run python -m gmtrack.scripts.audit_ground_clearance \
    --manifest logs/data_build/manifests/seed_stunts_grounded.json --threshold 0.003
```

Replay the reference with a zero-action policy to sanity-check a manifest:

```bash
uv run play <task-id> --agent zero
```

### 2. Stage I

Environment count and iteration budget are baked into the task configuration, so no
scale flags are needed:

```bash
uv run train <stage1-task-id>
```

### 3. Stratification

Use the same manifest Stage I trained on, otherwise `D_m ∪ D_c` no longer covers the
training distribution.

```bash
uv run python -m gmtrack.scripts.stratify \
    --checkpoint logs/rsl_rl/gmtrack_stage1/<run>/model_99999.pt \
    --manifest data/current/<stage1-manifest>.json
```

This produces `stratified.json`, `mastered.json`, `challenging.json` and
`stratification_report.json`. The four files share one provenance hash and are validated
as a set; hand-edited splits fail closed. A large `D_c` split usually means Stage I needs
more training.

### 4. Stage II

```bash
uv run train <stage2-task-id> \
    --agent.algorithm.base-checkpoint logs/rsl_rl/gmtrack_stage1/<run>/model_99999.pt
```

The same checkpoint warm-starts `π_θ` and freezes `π_ref`. A short gate run can be
checked before committing to the long job:

```bash
uv run python -m gmtrack.scripts.check_stage2_gate --run-dir logs/rsl_rl/gmtrack_stage2/<run>
```

### 5. Evaluate and aggregate

```bash
uv run python -m gmtrack.scripts.evaluate \
    --checkpoint <ckpt> --manifest <test-set>.json \
    --training-seed 42 --out logs/eval/seed42.json

uv run python -m gmtrack.scripts.aggregate_evaluations \
    --inputs logs/eval/seed4{2,3,4,5,6}.json --out logs/eval/five_seed_summary.json
```

`python -m gmtrack.scripts.webplay` opens a Viser viewer through the evaluation harness
for interactive inspection of a checkpoint; it loads policy weights only, never the
training motion sampler.

## Registered tasks

| Task ID | Description |
|---|---|
| `GMTrack-Stage1-Heading-Flat-Unitree-G1` | Stage I with heading feedback |
| `GMTrack-Stage2-Heading-Flat-Unitree-G1` | Stage II with heading feedback |
| `GMTrack-Stage1-Causal-Flat-Unitree-G1` | Stage I, past-only command window |
| `GMTrack-Stage1-Causal-Heading-Flat-Unitree-G1` | Stage I, past-only window with heading feedback |

`Causal` tasks use only current and past command tokens. `Heading` appends the 6D
robot-to-reference pelvis orientation error to each command token.

## Motion data

GMTrack expects retargeted 50 Hz Unitree G1 motions. A Stage-I manifest can combine
LAFAN1, AMASS, MotionDecode, BONES-SEED, and local motion captures as needed.

Motion payloads are ignored by git. Point the registry at a local dataset without
touching source code:

```bash
export GMTRACK_DATA=/abs/path/to/data              # dataset root
export GMTRACK_STAGE1_MANIFEST=/abs/path/to/stage1_complete_sequences.json
export GMTRACK_STRATIFIED_MANIFEST=/abs/path/to/stratified.json
export GMTRACK_MASTERED_MANIFEST=/abs/path/to/mastered.json
export GMTRACK_CHALLENGING_MANIFEST=/abs/path/to/challenging.json
```

Two conversion rules:

- **Body ordering follows MuJoCo's depth-first convention.** IsaacLab-derived converters
  silently track the wrong links, so use `prepare_motions` (or mjlab's `csv_to_npz`).
- **Clips retargeted from other skeletons are ground-aligned upward** against all
  29 active G1 collision geoms with a 3 mm clearance gate, re-differentiating root
  velocity afterwards.

## Evaluation protocol

A rollout fails when root height deviates from the reference by more than 0.2 m. Each
rollout runs from the reference's first frame to its end or to failure, and metrics are
Succ. / `E_MPJPE` (mm, root-relative) / `d_vel` (mm/frame) / `d_acc` (mm/frame²).

Keep final evaluation sets separate from `D_c`: `D_c` is part of the training split
and should not be used to report held-out performance.

## Checkpoints

Pretrained weights are published as GitHub Release assets rather than committed to git
history:

| Stage | Variant | Files |
|---|---|---|
| Stage I | no heading | [`stage1_no_heading_model_116500.pt`](https://github.com/cmjang/GMTrack/releases/download/checkpoint-stage1-baselines/stage1_no_heading_model_116500.pt) |
| Stage I | heading | [`stage1_heading_model_99999.pt`](https://github.com/cmjang/GMTrack/releases/download/checkpoint-stage1-baselines/stage1_heading_model_99999.pt) |
| Stage I | causal + heading | [`stage1_causal_heading_model_99999.pt`](https://github.com/cmjang/GMTrack/releases/download/checkpoint-stage1-baselines/stage1_causal_heading_model_99999.pt) |
| Stage II | no heading | [`model_17000.pt`](https://github.com/cmjang/GMTrack/releases/download/checkpoint-original-stage2-backflip/model_17000.pt), [`model_31500.pt`](https://github.com/cmjang/GMTrack/releases/download/checkpoint-original-stage2-backflip/model_31500.pt), [`model_32000.pt`](https://github.com/cmjang/GMTrack/releases/download/checkpoint-original-stage2-backflip/model_32000.pt) |

Checkpoints produced before the current round of alignment fixes are fine for inspection
but not for final numbers: retrain Stage I, rerun stratification, then retrain Stage II.

## Development

```bash
uv run pytest tests/ -v
uv run ruff check src tests
```

Conventions:

- mjlab style: 2-space indent, ruff `["E4", "E7", "E9", "F", "I", "B"]`.
- Subclass rsl-rl rather than editing `third_party/rsl_rl`.
- Prefer raising over defensive fallbacks. A silent clamp or `getattr(..., default)`
  hides the class of bug that is hardest to find here: a plausible but wrong reference
  frame.

### Project layout

| Path | Responsibility |
|---|---|
| `src/gmtrack/pace.py` | acquisition/consolidation environment split |
| `src/gmtrack/mdp/` | motion library, adaptive sampling, commands, observations, rewards, terminations, events |
| `src/gmtrack/envs/env_cfg.py` | full environment configuration |
| `src/gmtrack/rsl_rl/models.py` | three-branch encoder, causal attention, actor/critic |
| `src/gmtrack/rsl_rl/fsq.py` | finite scalar quantization bottleneck |
| `src/gmtrack/rsl_rl/ppo_pace.py` | PACE objectives and adaptive `λ_con` |
| `src/gmtrack/rsl_rl/storage.py` | STAR normalization and fragment resampling |
| `src/gmtrack/motion_grounding.py` | collision-geometry ground alignment |
| `src/gmtrack/scripts/` | data preparation, stratification, evaluation, viewers |

## References

GMTrack draws on ideas from the following related papers and open-source projects:

- Extreme-RGMT and RGMT
- [InstinctLab](https://github.com/project-instinct/InstinctLab) and
  [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl).
- [mjlab](https://github.com/mujocolab/mjlab) and
  [rsl-rl](https://github.com/leggedrobotics/rsl_rl), which this package builds on.

## License

No license has been chosen yet, so default copyright applies and no reuse rights are
granted. Open an issue if you need one.
