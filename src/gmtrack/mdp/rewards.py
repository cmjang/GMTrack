"""Reward terms for Extreme-RGMT.

The six motion-tracking terms are vendored verbatim from mjlab's tracking task
(``mjlab/tasks/tracking/mdp/rewards.py``) so they sit next to the ones we add and can
be edited without patching mjlab. They only touch the command's property surface,
which :class:`~gmtrack.mdp.commands.MultiMotionCommand` keeps compatible.

Paper Table I asks for four regularizers: action rate (-0.1), joint position limits
(-10.0), undesired contacts (-0.1) and feet slip (-0.1). The first two come from
``mjlab.envs.mdp.rewards``; the latter two are defined here because mjlab's versions
live in the velocity task and are gated on a twist command that tracking has no
concept of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from gmtrack.mdp.commands import MultiMotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MultiMotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MultiMotionCommand:
  return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


##
# Motion tracking (Table I, weights 0.5 / 0.5 / 1.0 / 1.0 / 1.0 / 1.0).
##


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _cmd(env, command_name)
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _cmd(env, command_name)
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, idx], command.robot_body_quat_w[:, idx]
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(command.body_lin_vel_w[:, idx] - command.robot_body_lin_vel_w[:, idx]),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(command.body_ang_vel_w[:, idx] - command.robot_body_ang_vel_w[:, idx]),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


##
# Regularizers.
##


def self_collision_cost(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
  """Penalize self-collisions (vendored from mjlab).

  With force history available (``history_length > 0`` on the sensor), counts
  substeps where any contact force exceeds *force_threshold*; otherwise falls back to
  the instantaneous ``found`` count.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


def undesired_contacts(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 1.0
) -> torch.Tensor:
  """Count non-foot bodies currently in contact with the ground (Table I, -0.1).

  Unlike mjlab's velocity-task ``illegal_contact`` termination, this is a graded
  penalty: highly dynamic references legitimately put knees/hands on the ground
  during kip-ups and cartwheels, so contact must be discouraged rather than fatal.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).sum(dim=-1).float()
  assert data.found is not None
  return (data.found > 0).sum(dim=-1).float()


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize squared horizontal foot velocity while in contact (Table I, -0.1).

  mjlab's velocity-task ``feet_slip`` gates on the commanded twist magnitude. Motion
  tracking has no twist command, and the reference itself dictates when the feet
  should be planted, so the penalty applies unconditionally.

  Uses foot *body* velocity rather than site velocity so it does not depend on the
  G1 XML declaring foot sites.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]  # [B, N, 2]
  return torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * in_contact, dim=1)
