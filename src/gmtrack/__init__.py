"""Extreme-RGMT: continual learning of highly dynamic humanoid skills.

Reproduction of arXiv:2607.20110 on top of mjlab. Importing this package registers
every task with mjlab's registry; mjlab auto-imports it through the
``[project.entry-points."mjlab.tasks"]`` hook, so ``uv run train GMTrack-...`` works
without any code change inside mjlab.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from gmtrack.envs.env_cfg import RECOVERY_PROBABILITY, make_gmtrack_env_cfg
from gmtrack.rl_cfgs import (
  finetune_runner_cfg,
  no_fsq_runner_cfg,
  stage1_runner_cfg,
  stage2_runner_cfg,
  unified_encoder_runner_cfg,
)
from gmtrack.rsl_rl.runner import GMTrackOnPolicyRunner


def data_dir() -> Path:
  """Root of the motion dataset.

  Defaults to ``<repo>/data`` (this package is installed editable from ``src/``).
  Override with ``GMTRACK_DATA`` when the dataset lives elsewhere -- e.g. on scratch
  storage on the cluster.
  """
  env = os.environ.get("GMTRACK_DATA")
  if env:
    return Path(env)
  return Path(__file__).resolve().parents[2] / "data"


def _manifest_path(env_name: str, default_relative_path: str) -> str:
  """Resolve an override or a path relative to the configured data directory."""
  override = os.environ.get(env_name)
  if override:
    return str(Path(override).expanduser().resolve())
  return str(data_dir() / default_relative_path)


MANIFEST_STAGE1 = _manifest_path(
  "GMTRACK_STAGE1_MANIFEST",
  "current/stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json",
)
_STANDARD_SPLIT = "current/stage1-116500-final30-cartwheel-balanced-nofall-probe"
MANIFEST_STRATIFIED = _manifest_path(
  "GMTRACK_STRATIFIED_MANIFEST", f"{_STANDARD_SPLIT}/stratified.json"
)
MANIFEST_MASTERED = _manifest_path(
  "GMTRACK_MASTERED_MANIFEST", f"{_STANDARD_SPLIT}/mastered.json"
)
MANIFEST_CHALLENGING = _manifest_path(
  "GMTRACK_CHALLENGING_MANIFEST", f"{_STANDARD_SPLIT}/challenging.json"
)

_HEADING_SPLIT = "current/stage1-heading-99999-cartwheel20-formal-seed42"
MANIFEST_HEADING_STRATIFIED = _manifest_path(
  "GMTRACK_STRATIFIED_MANIFEST", f"{_HEADING_SPLIT}/stratified.json"
)
MANIFEST_HEADING_MASTERED = _manifest_path(
  "GMTRACK_MASTERED_MANIFEST", f"{_HEADING_SPLIT}/mastered.json"
)
MANIFEST_HEADING_CHALLENGING = _manifest_path(
  "GMTRACK_CHALLENGING_MANIFEST", f"{_HEADING_SPLIT}/challenging.json"
)


def _stage1_env(play: bool = False):
  return make_gmtrack_env_cfg(manifest=MANIFEST_STAGE1, play=play)


def _stage1_causal_env(play: bool = False):
  """Legacy 38D causal config retained for checkpoint audit; not registered."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    causal_online=True,
    sonic_foot_terminations=True,
  )


def _stage1_causal_heading_env(play: bool = False):
  """Past-only Stage I with the existing SONIC-style heading feedback."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    causal_online=True,
    heading_closed_loop=True,
    sonic_foot_terminations=True,
  )


def _stage1_causal_recovery_env(play: bool = False):
  """Legacy 38D causal recovery config retained for audit; not registered."""
  cfg = make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    causal_online=True,
    sonic_foot_terminations=True,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  if play:
    cfg.commands["motion"].recovery_probability = 0.0
  return cfg


def _stage1_causal_heading_recovery_env(play: bool = False):
  """Causal + Heading + foot guards + RGMT Sec. II-D fall recovery."""
  cfg = make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    causal_online=True,
    heading_closed_loop=True,
    sonic_foot_terminations=True,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  if play:
    cfg.commands["motion"].recovery_probability = 0.0
  return cfg


def _stage1_recovery_env(play: bool = False):
  """Stage I with RGMT Sec. II-D fall recovery (docs/recovery_proxy.md).

  This has to be its own task because ``recovery_probability`` decides at construction
  time whether the assistance-force event exists. The play config retains that event
  but disables random recovery resets by default; ``webplay --random-recovery-start``
  can then enable deterministic, unassisted recovery visualization safely.
  """
  cfg = make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  if play:
    cfg.commands["motion"].recovery_probability = 0.0
  return cfg


def _stage1_heading_env(play: bool = False):
  """Stage-I environment with SONIC-style root-heading feedback."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    heading_closed_loop=True,
  )


def _stage1_heading_recovery_env(play: bool = False):
  """Heading-aware Stage I with RGMT Sec. II-D fall recovery."""
  cfg = make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    heading_closed_loop=True,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  if play:
    cfg.commands["motion"].recovery_probability = 0.0
  return cfg


def _stage2_env(play: bool = False):
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STRATIFIED,
    acquisition_clips=MANIFEST_CHALLENGING,
    consolidation_clips=MANIFEST_MASTERED,
    acquisition_fraction=0.8,
    require_v1_stratification=True,
    play=play,
  )


def _stage2_causal_env(play: bool = False):
  """Legacy 38D causal Stage-II config retained for audit; not registered."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STRATIFIED,
    acquisition_clips=MANIFEST_CHALLENGING,
    consolidation_clips=MANIFEST_MASTERED,
    acquisition_fraction=0.8,
    require_v1_stratification=True,
    play=play,
    causal_online=True,
  )


def _stage2_heading_env(play: bool = False):
  """Stage-II environment paired with the heading-aware Stage-I policy."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_HEADING_STRATIFIED,
    acquisition_clips=MANIFEST_HEADING_CHALLENGING,
    consolidation_clips=MANIFEST_HEADING_MASTERED,
    acquisition_fraction=0.8,
    require_v1_stratification=True,
    play=play,
    heading_closed_loop=True,
  )


def _challenging_only_env(play: bool = False):
  """Every environment on the challenging set -- the Fine-Tuning baseline."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STRATIFIED,
    acquisition_clips=MANIFEST_CHALLENGING,
    require_v1_stratification=True,
    stratification_mastered_manifest=MANIFEST_MASTERED,
    stratification_challenging_manifest=MANIFEST_CHALLENGING,
    play=play,
  )


def _mixed_env(play: bool = False):
  """Whole post-stratification distribution without PACE role separation."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STRATIFIED,
    require_v1_stratification=True,
    stratification_mastered_manifest=MANIFEST_MASTERED,
    stratification_challenging_manifest=MANIFEST_CHALLENGING,
    play=play,
  )


##
# Stage I: generalist base policy.
##

register_mjlab_task(
  task_id="GMTrack-Stage1-Flat-Unitree-G1",
  env_cfg=_stage1_env(),
  play_env_cfg=_stage1_env(play=True),
  rl_cfg=stage1_runner_cfg(),
  runner_cls=GMTrackOnPolicyRunner,
)

# Causal task IDs always include closed-loop Heading. Do not add separate
# ``Causal-Heading`` aliases: the short causal names are the sole public entry points.
register_mjlab_task(
  task_id="GMTrack-Stage1-Causal-Flat-Unitree-G1",
  env_cfg=_stage1_causal_heading_env(),
  play_env_cfg=_stage1_causal_heading_env(play=True),
  rl_cfg=stage1_runner_cfg(
    causal_online=True,
    heading_closed_loop=True,
    experiment_name="gmtrack_stage1_causal_heading",
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage1-Causal-Recovery-Flat-Unitree-G1",
  env_cfg=_stage1_causal_heading_recovery_env(),
  play_env_cfg=_stage1_causal_heading_recovery_env(play=True),
  rl_cfg=stage1_runner_cfg(
    causal_online=True,
    heading_closed_loop=True,
    experiment_name="gmtrack_stage1_causal_heading_recovery",
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

# Ablation of the boundary bookkeeping: identical environment and identical past-only
# actor window, but the actor is not told which of its tokens are clamped padding.
# The environment is shared on purpose -- the critic and the reconstruction head keep
# their masks, so the only variable is what the actor knows.
register_mjlab_task(
  task_id="GMTrack-Stage1-CausalNoMask-Flat-Unitree-G1",
  env_cfg=_stage1_causal_heading_env(),
  play_env_cfg=_stage1_causal_heading_env(play=True),
  rl_cfg=stage1_runner_cfg(
    causal_online=True,
    heading_closed_loop=True,
    actor_boundary_masks=False,
    experiment_name="gmtrack_stage1_causal_heading_nomask",
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage1-Recovery-Flat-Unitree-G1",
  env_cfg=_stage1_recovery_env(),
  play_env_cfg=_stage1_recovery_env(play=True),
  rl_cfg=stage1_runner_cfg(experiment_name="gmtrack_stage1_recovery"),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage1-Heading-Flat-Unitree-G1",
  env_cfg=_stage1_heading_env(),
  play_env_cfg=_stage1_heading_env(play=True),
  rl_cfg=stage1_runner_cfg(heading_closed_loop=True),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage1-Heading-Recovery-Flat-Unitree-G1",
  env_cfg=_stage1_heading_recovery_env(),
  play_env_cfg=_stage1_heading_recovery_env(play=True),
  rl_cfg=stage1_runner_cfg(
    heading_closed_loop=True,
    experiment_name="gmtrack_stage1_heading_recovery",
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

##
# Stage II: PACE + STAR.
##

register_mjlab_task(
  task_id="GMTrack-Stage2-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(),
  runner_cls=GMTrackOnPolicyRunner,
)

# No causal Stage-II task is registered yet. Causal now means causal + Heading, and a
# valid Stage-II registration must point at a split regenerated from that exact 44D
# Stage-I checkpoint. The existing standard/Heading splits do not satisfy that
# provenance contract, so failing closed here is safer than exposing a mismatched task.

register_mjlab_task(
  task_id="GMTrack-Stage2-Heading-Flat-Unitree-G1",
  env_cfg=_stage2_heading_env(),
  play_env_cfg=_stage2_heading_env(play=True),
  rl_cfg=stage2_runner_cfg(heading_closed_loop=True),
  runner_cls=GMTrackOnPolicyRunner,
)

##
# Ablations (Fig. 7, Fig. 9, Table VIII).
##

register_mjlab_task(
  task_id="GMTrack-Stage2-NoStar-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(use_star=False, experiment_name="gmtrack_no_star"),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage2-NoCon-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(
    consolidation_enabled=False, experiment_name="gmtrack_no_con"
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage2-FixedLambda-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=stage2_runner_cfg(
    fixed_lambda_con=0.5, experiment_name="gmtrack_fixed_lambda"
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage2-UnifiedEnc-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=unified_encoder_runner_cfg(),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage2-NoFSQ-Flat-Unitree-G1",
  env_cfg=_stage2_env(),
  play_env_cfg=_stage2_env(play=True),
  rl_cfg=no_fsq_runner_cfg(),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Finetune-Flat-Unitree-G1",
  env_cfg=_challenging_only_env(),
  play_env_cfg=_challenging_only_env(play=True),
  rl_cfg=finetune_runner_cfg(),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-MixedTraining-Flat-Unitree-G1",
  # Mixed Training (Fig. 9): one undifferentiated pool of mastered + challenging
  # motions optimized with plain PPO -- no role split, no consolidation constraint.
  env_cfg=_mixed_env(),
  play_env_cfg=_mixed_env(play=True),
  rl_cfg=finetune_runner_cfg(),
  runner_cls=GMTrackOnPolicyRunner,
)
