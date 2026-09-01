# GMTrack

[![Paper](https://img.shields.io/badge/arXiv-2607.20110-b31b1b.svg)](https://arxiv.org/abs/2607.20110)
[![Built on mjlab](https://img.shields.io/badge/built%20on-mjlab-4c1.svg)](https://github.com/mujocolab/mjlab)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)

Unofficial reproduction of highly dynamic humanoid whole-body motion tracking on the
Unitree G1 (29 DoF), written against two public references:

- **Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust Generalist
  Humanoid Control** (arXiv:2607.20110) — the method: the two-stage pipeline, PACE,
  STAR, the encoder architecture, the reward and randomization tables.
- **SONIC** (and BeyondMimic / InstinctLab) — the implementation values the paper never
  publishes: parallel environment count, iteration budget, simulation rate, adaptive
  sampling constants, FSQ levels. Every such value is labelled in code with the release
  it came from, so nothing borrowed is ever presented as a paper number.

> **GMTrack is not the official implementation.** It is an independent reimplementation
> from the papers, not affiliated with or endorsed by any of the original authors, and it
> carries its own name so it is never mistaken for their release. Numbers produced here
> are ours, not theirs.

Extreme-RGMT extends BeyondMimic, and [mjlab](https://github.com/mujocolab/mjlab)
already ships a BeyondMimic reimplementation under `mjlab/tasks/tracking/`. This
repository is therefore an **external mjlab task package**: it registers its own tasks
through mjlab's entry-point hook and subclasses rsl-rl instead of forking either
project.

## Method

```
D — all motion data (paper Table IV)
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

Two mechanisms carry Stage II:

- **PACE** — asymmetric acquisition/consolidation environment roles plus a
  progress-adaptive regularizer `λ_con` toward the frozen base policy, so learning
  new highly dynamic skills does not erase mastered ones.
- **STAR** — advantage-prioritized resampling of trajectory fragments drawn from the
  hardest temporal bins, which compensates for the sparse and failure-heavy samples
  produced by extreme motions.

The policy itself is a three-branch encoder (proprioceptive state, past actions,
command window) feeding a causal attention encoder with a finite-scalar-quantization
bottleneck; the actor consumes only deployable observations.

## Status

The implementation is complete and covered by unit tests, and a first end-to-end
formal chain (Stage I → stratification → Stage II) has finished on the cluster.
**No reproduction numbers are claimed yet** — the paper reports mean ± std over five
independently trained seeds on four test sets, and that evaluation sweep is still
outstanding.

Checkpoints produced before the current paper-alignment fixes must not be used for
final results: retrain Stage I, rerun stratification, then retrain Stage II.

## Installation

Requires Linux x86_64 with an NVIDIA GPU — `mujoco-warp` has no CPU path, so a CPU
install cannot train. torch is pinned to the CUDA 12.8 index unconditionally.

```bash
uv sync
uv run list-envs | grep GMTrack
```

`third_party/rsl_rl` is a local editable checkout of `rsl-rl-lib` 5.4.0, the version
mjlab pins exactly. It is never patched — every modification lives in
`src/gmtrack/rsl_rl/` and is injected through rsl-rl's `"module:Class"` resolution.

`uv run` re-syncs the environment on every invocation; use `.venv/bin/python -m ...`
directly to skip that.

## Pipeline

### 1. Prepare motions (local)

`prepare_motions` writes one complete sequence per source CSV — no 10 s slicing
happens here.

```bash
uv run python -m gmtrack.scripts.prepare_motions \
    --input-dir data/datasets/raw/lafan1 --source lafan1 \
    --input-format mjlab --input-fps 30 \
    --output-dir data/datasets/stage1_full/lafan1 \
    --manifest logs/data_build/manifests/lafan1_full.json
```

High-dynamic proxy clips are additionally aligned against the G1 collision geometry
and must pass the clearance gate before entering a manifest:

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
uv run play GMTrack-Stage1-Flat-Unitree-G1 --agent zero
```

### 2. Stage I (cluster)

Environment count and iteration budget are baked into the task configuration, so no
scale flags are needed:

```bash
uv run train GMTrack-Stage1-Flat-Unitree-G1
```

### 3. Stratification (single GPU)

Must use the same manifest Stage I trained on, otherwise `D_m ∪ D_c` no longer covers
the training distribution.

```bash
uv run python -m gmtrack.scripts.stratify \
    --checkpoint logs/rsl_rl/gmtrack_stage1/<run>/model_99999.pt \
    --manifest data/current/<stage1-manifest>.json
```

This produces `stratified.json`, `mastered.json`, `challenging.json` and
`stratification_report.json`. The four files share one provenance hash and are
validated as a set; hand-edited splits fail closed. Against paper Table V, `D_c`
should land around 10% of the corpus — close to half means Stage I is undertrained.

### 4. Stage II (cluster)

```bash
uv run train GMTrack-Stage2-Flat-Unitree-G1 \
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

To inspect a checkpoint interactively, `python -m gmtrack.scripts.webplay` opens a
Viser viewer through the evaluation harness — it loads policy weights only, never the
training motion sampler.

Cluster jobs live under `scripts/slurm/`. The scripts take every machine-specific
path from the environment (`GMTRACK_PROJECT_DIR`, `GMTRACK_DATA`, `CONDA_ENV_PREFIX`)
rather than hard-coding a cluster layout.

## Registered tasks

| Task ID | Role |
|---|---|
| `GMTrack-Stage1-Flat-Unitree-G1` | Stage I generalist base policy |
| `GMTrack-Stage1-Causal-Flat-Unitree-G1` | Zero-lookahead Stage I; Heading is built in by default (44D tokens, yaw reset), with pelvis-realigned foot XY/Z terminations (0.20/0.15 m) |
| `GMTrack-Stage1-Causal-Recovery-Flat-Unitree-G1` | Default causal task plus 15% RGMT fall-recovery resets/assist/shield; Heading remains built in |
| `GMTrack-Stage2-Flat-Unitree-G1` | Stage II with PACE + STAR (the method) |
| `GMTrack-Stage1-Heading-Flat-Unitree-G1` | Stage I with closed-loop root heading (44D tokens) |
| `GMTrack-Stage2-Heading-Flat-Unitree-G1` | Stage II counterpart of the heading family |
| `GMTrack-Stage1-Recovery-Flat-Unitree-G1` | Stage I plus the RGMT fall-recovery protocol |
| `GMTrack-Stage1-Heading-Recovery-Flat-Unitree-G1` | Both of the above |
| `GMTrack-Stage2-NoStar-Flat-Unitree-G1` | Ablation: no STAR (Fig. 7–8, Table VII) |
| `GMTrack-Stage2-NoCon-Flat-Unitree-G1` | Ablation: no consolidation loss (Fig. 9) |
| `GMTrack-Stage2-FixedLambda-Flat-Unitree-G1` | Ablation: `λ_con ≡ 0.5` (Fig. 9) |
| `GMTrack-Stage2-UnifiedEnc-Flat-Unitree-G1` | Ablation: unified encoder (Table VIII) |
| `GMTrack-Stage2-NoFSQ-Flat-Unitree-G1` | Ablation: no FSQ bottleneck (Table VIII) |
| `GMTrack-Finetune-Flat-Unitree-G1` | Baseline: fine-tune on `D_c` only |
| `GMTrack-MixedTraining-Flat-Unitree-G1` | Baseline: one undifferentiated pool, no roles |

The heading family appends the 6D robot-to-reference pelvis orientation error to every
command token. Causal task names always include this feedback and use 44D tokens; no
separate `Causal-Heading` aliases are registered. Historical 38D causal checkpoints
are retained only for audit and are incompatible with the current causal task names.

The explicit-intent causal family keeps the same past-only external inputs, but appends
the deterministic mean of its history-conditioned intent distribution to the actor
trunk. The stochastic sample, future decoder, and future targets remain training-only.
Its enlarged actor trunk is checkpoint-incompatible with the ordinary causal family.

## Motion data

The paper trains Stage I on 3.096 h of retargeted 50 Hz motion:

| Source | Paper duration | Availability |
|---|---|---|
| LAFAN1 | 2.444 h | public — [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), drop the CSVs into `data/datasets/raw/lafan1/` |
| AMASS | 0.511 h | needs your own retargeting to G1 |
| In-house Xsens captures | 0.141 h | not public |

Because the Xsens high-dynamic slice is unavailable, the registered Stage-I task
defaults to a duration-matched **proxy** built from public high-dynamic sources
(screened MotionDecode and BONES-SEED takes plus representative cartwheels). Any
experiment report using it must say so — it is not the paper's data distribution.
Motion payloads are ignored by git; see [`data/README.md`](data/README.md) for the
expected layout, and point the registry elsewhere without touching source code:

```bash
export GMTRACK_DATA=/abs/path/to/data              # dataset root
export GMTRACK_STAGE1_MANIFEST=/abs/path/to/stage1_complete_sequences.json
export GMTRACK_STRATIFIED_MANIFEST=/abs/path/to/stratified.json
export GMTRACK_MASTERED_MANIFEST=/abs/path/to/mastered.json
export GMTRACK_CHALLENGING_MANIFEST=/abs/path/to/challenging.json
```

Two conversion rules matter. Body ordering follows MuJoCo's depth-first convention,
so IsaacLab-derived converters silently track the wrong links — use
`prepare_motions` (or mjlab's `csv_to_npz`) only. And proxy clips retargeted from
other skeletons are ground-aligned upward against all 29 active G1 collision geoms
with a 3 mm clearance gate, re-differentiating root velocity afterwards; that
threshold is a local data-cleaning choice, not a paper parameter.

## Evaluation protocol

Following Sec. VI-A, the only failure criterion is root height deviating from the
reference by more than 0.2 m. Each rollout runs from the reference's first frame to
its end or to failure, and metrics are Succ. / `E_MPJPE` (mm, root-relative) /
`d_vel` (mm/frame) / `d_acc` (mm/frame²), averaged over five training seeds.

The paper's four test sets are all distinct from `D_c` — `D_c` is a training set and
numbers on it cannot be compared to Table VI. Its reported values, as
Succ. / `E_MPJPE`:

| Category | Test set | Stage I | Full (Stage II) |
|---|---|---|---|
| Generalist | In-source Motion (held out from LAFAN1 + AMASS) | 99.54% / 40.07 mm | 99.76% / 40.79 mm |
| Generalist | Unseen Motion (independent video reconstructions) | 95.13% / 45.80 mm | 96.68% / 46.91 mm |
| Specialist | XtremeMotion (public OmniXtreme high-dynamic set) | 21.42% / 46.72 mm | 100.00% / 40.18 mm |
| Specialist | AMASS Challenging (hard AMASS motions, direct retarget) | 18.18% / 55.17 mm | 90.91% / 46.39 mm |

The specialist gap is the entire point of Stage II.

## Deviations and assumptions

Everything the paper specifies is followed as written. Where it is silent, the value
comes from the reference implementations on this line of work and is marked as such —
never presented as a paper value.

| Item | Value used | Source |
|---|---|---|
| Parallel environments | 4096 (4 GPUs × 1024) | BeyondMimic / SONIC convention |
| Iterations | 100k (Stage I), 50k (Stage II) | SONIC-scale budget |
| Simulation / policy rate | 200 Hz / 50 Hz, decimation 4 | BeyondMimic / SONIC |
| Adaptive sampling | bin 1 s, α = 0.001, ε_u = 0.1, `c_max = mean × 200` | BeyondMimic + SONIC |
| FSQ levels | 32 per dimension | SONIC's matching 2 × 32 tokenizer |
| Attention block internals | 4 heads, pre-LN residual blocks, sinusoidal PE | RGMT (arXiv:2601.23080), Eq. 9–11/14–15 |
| Fall recovery | RGMT Sec. II-D protocol, off by default | [`docs/recovery_proxy.md`](docs/recovery_proxy.md) |
| Ground alignment | ≥ 3 mm clearance, 0.3 s smoothing | local data cleaning |
| Closed-loop heading | opt-in `-Heading-` task family | local extension |

A few design constraints are easy to break by accident and are worth knowing before
editing:

- **No running-statistic observation normalization** on the policy (Sec. IV-A); the
  three branches each use LayerNorm. The critic keeps its own normalizer.
- **Actions are residuals on the reference joint pose** `q_tar = q_ref + a_t` (Eq. 3),
  not on a constant standing pose.
- **The critic sees a single current reference token**, not the 21-token command
  window — feeding the window changes value estimates, GAE, and STAR statistics.
- **Domain randomization scope**: all environments in Stage I, but only the
  acquisition group in Stage II (Sec. IV-B2).
- **Role assignment has exactly one authority**, `pace.py::pace_env_split`; if the
  environment side and the storage side ever disagree, the run silently degrades into
  the Mixed Training ablation.

Every paper constant is tagged in code with the table or equation it comes from, and
every place the paper is underspecified carries a comment naming the source of the
value used instead (BeyondMimic, SONIC, RGMT, or a local choice).

## Development

```bash
uv run pytest tests/ -v
uv run ruff check src tests
```

Style follows mjlab: 2-space indent, ruff `["E4", "E7", "E9", "F", "I", "B"]`.
Subclass rsl-rl rather than editing `third_party/rsl_rl`, and prefer raising over
defensive fallbacks — a silent clamp or `getattr(..., default)` hides exactly the
class of bug that is hardest to find here, a plausible but wrong reference frame.

### Code map

| Path | Responsibility |
|---|---|
| `src/gmtrack/pace.py` | acquisition/consolidation environment split |
| `src/gmtrack/mdp/` | motion library, adaptive sampling, commands, observations, rewards, terminations, events |
| `src/gmtrack/envs/env_cfg.py` | full environment configuration (paper Tables I–II) |
| `src/gmtrack/rsl_rl/models.py` | three-branch encoder, causal attention, actor/critic |
| `src/gmtrack/rsl_rl/fsq.py` | finite scalar quantization bottleneck |
| `src/gmtrack/rsl_rl/ppo_pace.py` | PACE objectives and adaptive `λ_con` |
| `src/gmtrack/rsl_rl/storage.py` | STAR normalization and fragment resampling |
| `src/gmtrack/motion_grounding.py` | collision-geometry ground alignment |
| `src/gmtrack/scripts/` | data preparation, stratification, evaluation, viewers |

## Citation

If you use this code, please cite the original paper (Ma et al., 2026; [project
page](https://zeonsunlightyu.github.io/Extreme-RGMT.github.io/)):

```bibtex
@article{ma2026extremergmt,
  title   = {Extreme-RGMT: Continual Learning of Highly Dynamic Skills for
             Robust Generalist Humanoid Control},
  author  = {Ma, Yubiao and Yu, Han and Guo, Kai and Lv, Changtai and
             Mao, Zhengquan and Xing, Boyang and Ren, Xuemei and Zheng, Dongdong},
  journal = {arXiv preprint arXiv:2607.20110},
  year    = {2026}
}
```

This repository reproduces revision **v1** specifically (the PDF is not
redistributed here; SHA256
`55cca5c02f16c659e4ab3baf08d9ad1fb69865f37cfba084958ade0911cf51fe`). Details the
paper omits are taken from its predecessor RGMT (arXiv:2601.23080) and are labelled
as such throughout.

## Acknowledgements

Built on [mjlab](https://github.com/mujocolab/mjlab) and
[rsl-rl](https://github.com/leggedrobotics/rsl_rl). Unpublished low-level choices —
action scaling, actuator parameters, collision primitives, push ranges, tokenizer
width — are resolved against the public BeyondMimic/InstinctLab and SONIC releases;
neither contains RGMT or Extreme-RGMT source, so the two papers remain the only
authority for PACE, STAR, the encoder, and the randomization ranges.

## License

Not yet specified — no reuse rights are granted yet. Open an issue if you need one.
