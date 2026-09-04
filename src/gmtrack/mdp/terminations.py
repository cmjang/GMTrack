"""Termination terms for GMTrack."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from gmtrack.mdp.rewards import _cmd, _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


_PHYSICS_STATE_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "sensordata",
)


def nonfinite_physics_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate worlds before a bad simulator state reaches the policy.

  ``proprio_hist`` is built directly from the floating-base quaternion, IMU angular
  velocity, joint positions and joint velocities.  A non-finite value in any of
  those channels therefore originates in MuJoCo's state, not in the bounded
  observation noise.  The ordinary tracking terminations cannot catch it because a
  comparison such as ``NaN > threshold`` is false.

  This guard runs after the decimated physics steps and before auto-reset.  Marking
  the affected world as terminated makes mjlab reset it before ``sim.sense()`` and
  observation-history update, so one corrupt world cannot kill every distributed
  rank at the next adaptive-sampler collective.  mjlab's reward manager already
  maps a corrupt terminal transition's non-finite reward terms to zero.
  """
  bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for field in _PHYSICS_STATE_FIELDS:
    value = getattr(env.sim.data, field, None)
    if value is None:
      continue
    bad |= ~torch.isfinite(value).reshape(env.num_envs, -1).all(dim=1)
  return bad


def motion_sequence_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Time out when the current reference reaches its final frame.

  Stage I samples start bins across complete source sequences. An episode must end at
  the selected sequence boundary instead of letting ``MultiMotionCommand`` jump to a
  different sequence inside the same return trajectory.
  """
  command = _cmd(env, command_name)
  return command.time_steps >= command.lib.clip_len[command.motion_ids] - 1


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = _cmd(env, command_name)
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """RGMT Sec. II-D instability check: "insufficient base height".

  A symmetric deviation from the reference height, identical to mjlab's BeyondMimic
  port -- both papers state they follow BeyondMimic ([16]) and neither gives
  pseudocode. RGMT's prose ("insufficient", and "abnormally low" for the body-link
  check) reads one-directional, but that wording is not enough to depart from the
  implementation they inherited. An absolute floor is excluded outright: over the
  Stage-I corpus the lowest tracked end-effector sits at ground level in 95% of
  reference frames (feet), so no absolute threshold is definable there.
  """
  command = _cmd(env, command_name)
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = _cmd(env, command_name)
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )
  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )
  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Terminate on anchor-realigned full-XYZ body-position error.

  ``MultiMotionCommand.update_relative_body_poses`` translates the reference to the
  robot's current anchor XY and aligns its yaw before populating
  ``body_pos_relative_w``.  This therefore measures each selected body's placement
  relative to the current robot pelvis rather than absolute world-trajectory drift.
  """
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx], dim=-1
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_xy(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Terminate on pelvis-XY/yaw-realigned horizontal body-placement error."""
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error_xy = torch.norm(
    command.body_pos_relative_w[:, idx, :2] - command.robot_body_pos_w[:, idx, :2],
    dim=-1,
  )
  return torch.any(error_xy > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """RGMT Sec. II-D check: "abnormally low height of key body links".

  Symmetric, identical to mjlab's BeyondMimic port -- see ``bad_anchor_pos_z_only``.
  """
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, idx, -1] - command.robot_body_pos_w[:, idx, -1]
  )
  return torch.any(error > threshold, dim=-1)


# -- fall-recovery shields (RGMT, arXiv:2601.23080v1, Sec. II-D) --------------
#
# Recovery environments start from a deliberate fallen pose, so the instability
# checks are suspended for the first ``recovery_window_s`` seconds of the episode
# to let the policy stand up and re-stabilize. Once the window elapses the checks
# re-engage unchanged -- RGMT's "if the robot fails to recover within this window,
# the episode is terminated". For non-recovery environments (and any config with
# ``recovery_probability=0``) the shield is a no-op and these reduce exactly to
# their base terms.


def bad_anchor_pos_z_only_outside_recovery(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = _cmd(env, command_name)
  failing = bad_anchor_pos_z_only(env, command_name, threshold)
  return failing & ~command.in_recovery_window


def bad_anchor_ori_outside_recovery(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  failing = bad_anchor_ori(env, asset_cfg, command_name, threshold)
  return failing & ~command.in_recovery_window


def bad_motion_body_pos_xy_outside_recovery(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Recovery-shielded horizontal body-placement termination."""
  command = _cmd(env, command_name)
  failing = bad_motion_body_pos_xy(env, command_name, threshold, body_names)
  return failing & ~command.in_recovery_window


def bad_motion_body_pos_z_only_outside_recovery(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  failing = bad_motion_body_pos_z_only(env, command_name, threshold, body_names)
  return failing & ~command.in_recovery_window
