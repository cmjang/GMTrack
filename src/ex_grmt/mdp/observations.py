"""Observation terms for Extreme-RGMT.

The first four functions are vendored from mjlab's tracking task
(``mjlab/tasks/tracking/mdp/observations.py``) with the command type swapped for
:class:`~ex_grmt.mdp.commands.MultiMotionCommand`. The rest are new and back the
paper's command encoder (Sec. IV-A) and STAR (Sec. V-B).

Note on the proprioceptive history: mjlab's observation pipeline is
``compute -> noise -> clip -> scale -> delay -> history``, so each buffered frame
carries its own independent noise draw. That is what we want -- the history encoder
must see realistically noisy past frames, not one noise sample smeared across time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from ex_grmt.mdp.commands import MultiMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MultiMotionCommand:
  return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


##
# Reference-relative observations (vendored from mjlab).
##


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _cmd(env, command_name)
  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _cmd(env, command_name)
  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _cmd(env, command_name)
  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _cmd(env, command_name)
  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


##
# Extreme-RGMT: command encoder input.
##


def motion_command_window(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Flattened local reference window, shape ``(N, (2L+1) * (9 + J))``.

  Kept 2-D on purpose. rsl-rl's ``MLPModel._get_obs_dim`` rejects observation groups
  with more than two dimensions and ``RolloutStorage`` is happiest with flat rows, so
  the token axis is restored inside the policy
  (:class:`ex_grmt.rsl_rl.models.ExGRMTActor`) rather than here.
  """
  return _cmd(env, command_name).command_window().flatten(1)


def motion_ref_root_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference base height, ``(N, 1)``. Privileged -- critic only (paper Sec. III-B)."""
  command = _cmd(env, command_name)
  return command.body_pos_w[:, 0, 2:3] - env.scene.env_origins[:, 2:3]


##
# Extreme-RGMT: STAR metadata.
##


def motion_star_meta(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """``(N, 2)``: STAR difficulty weight ``w_t`` and the transition's bin id.

  This is not a policy input. It rides along as its own observation group purely so
  that ``RolloutStorage`` -- which stores the whole observation TensorDict -- carries
  it into the PPO update, where
  :class:`ex_grmt.rsl_rl.storage.StarRolloutStorage` needs it per transition. Keep
  this group out of every model's ``obs_groups`` entry.
  """
  return _cmd(env, command_name).star_meta()
