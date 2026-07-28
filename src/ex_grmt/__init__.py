"""Extreme-RGMT: continual learning of highly dynamic humanoid skills.

Reproduction of arXiv:2607.20110 on top of mjlab. Importing this package registers
every task with mjlab's registry; mjlab auto-imports it through the
``[project.entry-points."mjlab.tasks"]`` hook, so ``uv run train ExGRMT-...`` works
without any code change inside mjlab.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from ex_grmt.envs.env_cfg import make_ex_grmt_env_cfg
from ex_grmt.rl_cfgs import (
  finetune_runner_cfg,
  no_fsq_runner_cfg,
  stage1_runner_cfg,
  stage2_runner_cfg,
  unified_encoder_runner_cfg,
)
from ex_grmt.rsl_rl.runner import ExGRMTOnPolicyRunner


def data_dir() -> Path:
  """Root of the motion dataset.

  Defaults to ``<repo>/data`` (this package is installed editable from ``src/``).
  Override with ``EX_GRMT_DATA`` when the dataset lives elsewhere -- e.g. on scratch
  storage on the cluster.
  """
  env = os.environ.get("EX_GRMT_DATA")
  if env:
    return Path(env)
  return Path(__file__).resolve().parents[2] / "data"


MANIFEST_ALL = str(data_dir() / "manifests" / "all.json")
MANIFEST_MASTERED = str(data_dir() / "manifests" / "mastered.json")
MANIFEST_CHALLENGING = str(data_dir() / "manifests" / "challenging.json")


def _stage1_env(play: bool = False):
  return make_ex_grmt_env_cfg(manifest=MANIFEST_ALL, play=play)


def _stage2_env(play: bool = False):
  return make_ex_grmt_env_cfg(
    manifest=MANIFEST_ALL,
    acquisition_clips=MANIFEST_CHALLENGING,
    consolidation_clips=MANIFEST_MASTERED,
    acquisition_fraction=0.8,
    play=play,
  )


def _challenging_only_env(play: bool = False):
  """Every environment on the challenging set -- the Fine-Tuning baseline."""
  return make_ex_grmt_env_cfg(
    manifest=MANIFEST_ALL,
    acquisition_clips=MANIFEST_CHALLENGING,
    play=play,
  )


##
# Stage I: generalist base policy.
##

register_mjlab_task(
  task_id="ExGRMT-Stage1-Flat-Unitree-G1",
  env_cfg=_stage1_env(),
  play_env_cfg=_stage1_env(play=True),
  rl_cfg=stage1_runner_cfg(),
  runner_cls=ExGRMTOnPolicyRunner,
)

##
# Stage II: PACE + STAR.
##

register_mjlab_task(
  task_id="ExGRMT-Stage2-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(),
  runner_cls=ExGRMTOnPolicyRunner,
)

##
# Ablations (Fig. 7, Fig. 9, Table VIII).
##

register_mjlab_task(
  task_id="ExGRMT-Stage2-NoStar-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(use_star=False, experiment_name="ex_grmt_no_star"),
  runner_cls=ExGRMTOnPolicyRunner,
)

register_mjlab_task(
  task_id="ExGRMT-Stage2-NoCon-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(
    consolidation_enabled=False, experiment_name="ex_grmt_no_con"
  ),
  runner_cls=ExGRMTOnPolicyRunner,
)

register_mjlab_task(
  task_id="ExGRMT-Stage2-FixedLambda-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(
    fixed_lambda_con=0.5, experiment_name="ex_grmt_fixed_lambda"
  ),
  runner_cls=ExGRMTOnPolicyRunner,
)

register_mjlab_task(
  task_id="ExGRMT-Stage2-UnifiedEnc-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=unified_encoder_runner_cfg(),
  runner_cls=ExGRMTOnPolicyRunner,
)

register_mjlab_task(
  task_id="ExGRMT-Stage2-NoFSQ-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=no_fsq_runner_cfg(),
  runner_cls=ExGRMTOnPolicyRunner,
)

register_mjlab_task(
  task_id="ExGRMT-Finetune-Flat-Unitree-G1",
  env_cfg=_challenging_only_env(),
  play_env_cfg=_challenging_only_env(play=True),
  rl_cfg=finetune_runner_cfg(),
  runner_cls=ExGRMTOnPolicyRunner,
)

register_mjlab_task(
  task_id="ExGRMT-MixedTraining-Flat-Unitree-G1",
  # Mixed Training (Fig. 9): one undifferentiated pool of mastered + challenging
  # motions optimized with plain PPO -- no role split, no consolidation constraint.
  env_cfg=_stage1_env(),
  play_env_cfg=_stage1_env(play=True),
  rl_cfg=finetune_runner_cfg(),
  runner_cls=ExGRMTOnPolicyRunner,
)
