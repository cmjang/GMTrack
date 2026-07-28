# Ex-GRMT

Reproduction of **Extreme-RGMT: Continual Learning of Highly Dynamic Skills for Robust
Generalist Humanoid Control** ([arXiv:2607.20110](https://arxiv.org/abs/2607.20110)) on
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
uv sync --extra cu128       # or --extra cpu
uv run list-envs | grep ExGRMT
```

`third_party/rsl_rl` is a local editable checkout of `rsl-rl-lib` v5.4.0 (the version
mjlab pins). It is not patched — all modifications live in `src/ex_grmt/rsl_rl/` and
plug in through rsl-rl's `class_name` mechanism.

## Quick start

```bash
# 1. Build the motion library from retargeted CSVs
uv run python -m ex_grmt.scripts.prepare_motions \
    --input-dir data/raw/lafan1 --source lafan1 --input-fps 30

# 2. Stage I
uv run train ExGRMT-Stage1-Flat-Unitree-G1 --env.scene.num-envs 4096 --agent.max-iterations 30000

# 3. Split into mastered / challenging sets
uv run python -m ex_grmt.scripts.stratify --checkpoint logs/rsl_rl/ex_grmt_stage1/<run>/model_29999.pt

# 4. Stage II
uv run train ExGRMT-Stage2-Flat-Unitree-G1 \
    --agent.algorithm.base-checkpoint logs/rsl_rl/ex_grmt_stage1/<run>/model_29999.pt

# 5. Evaluate
uv run python -m ex_grmt.scripts.evaluate --checkpoint <ckpt> --manifest data/manifests/challenging.json
```

Motion data comes from
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
(already retargeted to the G1). Place the CSVs under `data/raw/lafan1/`.

See [CLAUDE.md](CLAUDE.md) for the paper→code map, the full constant tables, the list
of assumptions where the paper is underspecified, and the known gotchas.

## Status

Implementation complete; validated at small scale. Paper-scale numbers require
cluster training (4096 envs × 30k iterations ≈ 3B environment steps for Stage I
alone); see `scripts/slurm/train.sbatch`.
