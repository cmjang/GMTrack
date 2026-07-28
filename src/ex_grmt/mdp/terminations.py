"""Termination terms for Extreme-RGMT.

Vendored from mjlab's tracking task (``mjlab/tasks/tracking/mdp/terminations.py``)
so they can be edited in place. Behaviour is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ex_grmt.mdp.rewards import _cmd, _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


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
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, idx] - command.robot_body_pos_w[:, idx], dim=-1
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _cmd(env, command_name)
  idx = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, idx, -1] - command.robot_body_pos_w[:, idx, -1]
  )
  return torch.any(error > threshold, dim=-1)
