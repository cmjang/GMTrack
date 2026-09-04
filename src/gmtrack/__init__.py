"""GMTrack: continual learning of highly dynamic humanoid skills.

Built on top of mjlab. Importing this package registers every task with mjlab's
registry; mjlab auto-imports it through the
``[project.entry-points."mjlab.tasks"]`` hook, so ``uv run train GMTrack-...`` works
without any code change inside mjlab.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from gmtrack.envs.env_cfg import make_gmtrack_env_cfg
from gmtrack.rl_cfgs import stage1_runner_cfg, stage2_runner_cfg
from gmtrack.rsl_rl.runner import GMTrackOnPolicyRunner


def data_dir() -> Path:
  """Root of the motion dataset.

  Defaults to ``<repo>/data`` (this package is installed editable from ``src/``).
  Override with ``GMTRACK_DATA`` when the dataset lives elsewhere.
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


def _stage1_heading_env(play: bool = False):
  """Stage I with closed-loop heading feedback and the original command window."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    heading_closed_loop=True,
  )


def _stage1_causal_env(play: bool = False):
  """Past-only Stage I without heading feedback."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    causal_online=True,
    sonic_foot_terminations=True,
  )


def _stage1_causal_heading_env(play: bool = False):
  """Past-only Stage I with closed-loop heading feedback."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STAGE1,
    play=play,
    causal_online=True,
    heading_closed_loop=True,
    sonic_foot_terminations=True,
  )


def _stage2_env(play: bool = False):
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_STRATIFIED,
    acquisition_clips=MANIFEST_CHALLENGING,
    consolidation_clips=MANIFEST_MASTERED,
    acquisition_fraction=0.8,
    require_v1_stratification=True,
    play=play,
  )


def _stage2_heading_env(play: bool = False):
  """Stage II paired with the heading-aware Stage I policy."""
  return make_gmtrack_env_cfg(
    manifest=MANIFEST_HEADING_STRATIFIED,
    acquisition_clips=MANIFEST_HEADING_CHALLENGING,
    consolidation_clips=MANIFEST_HEADING_MASTERED,
    acquisition_fraction=0.8,
    require_v1_stratification=True,
    play=play,
    heading_closed_loop=True,
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

register_mjlab_task(
  task_id="GMTrack-Stage1-Heading-Flat-Unitree-G1",
  env_cfg=_stage1_heading_env(),
  play_env_cfg=_stage1_heading_env(play=True),
  rl_cfg=stage1_runner_cfg(heading_closed_loop=True),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage1-Causal-Flat-Unitree-G1",
  env_cfg=_stage1_causal_env(),
  play_env_cfg=_stage1_causal_env(play=True),
  rl_cfg=stage1_runner_cfg(
    causal_online=True,
    experiment_name="gmtrack_stage1_causal",
  ),
  runner_cls=GMTrackOnPolicyRunner,
)

register_mjlab_task(
  task_id="GMTrack-Stage1-Causal-Heading-Flat-Unitree-G1",
  env_cfg=_stage1_causal_heading_env(),
  play_env_cfg=_stage1_causal_heading_env(play=True),
  rl_cfg=stage1_runner_cfg(
    causal_online=True,
    heading_closed_loop=True,
    experiment_name="gmtrack_stage1_causal_heading",
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

register_mjlab_task(
  task_id="GMTrack-Stage2-Heading-Flat-Unitree-G1",
  env_cfg=_stage2_heading_env(),
  play_env_cfg=_stage2_heading_env(play=True),
  rl_cfg=stage2_runner_cfg(heading_closed_loop=True),
  runner_cls=GMTrackOnPolicyRunner,
)
