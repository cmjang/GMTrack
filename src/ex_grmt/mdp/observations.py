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
from mjlab.envs import mdp as mj_mdp
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from ex_grmt.mdp.commands import MultiMotionCommand
from ex_grmt.pace import pace_env_split

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> MultiMotionCommand:
  return cast(MultiMotionCommand, env.command_manager.get_term(command_name))


def acquisition_env_mask(
  num_envs: int,
  acquisition_fraction: float | None,
  device: torch.device | str,
) -> torch.Tensor:
  """Rows that receive the paper's training perturbation protocol.

  Stage I has no role split and perturbs every environment.  In Stage II the PACE
  split is index-based, so the first ``xi * N`` rows are acquisition environments and
  only those rows may receive observation/reference corruption (Sec. IV-B2).
  """
  split = (
    num_envs
    if acquisition_fraction is None
    else pace_env_split(acquisition_fraction, num_envs)
  )
  return torch.arange(num_envs, device=device) < split


def _add_acquisition_uniform_noise(
  data: torch.Tensor,
  magnitude: float | tuple[float, ...],
  acquisition_fraction: float | None,
  enabled: bool,
) -> torch.Tensor:
  """Apply symmetric uniform noise only to acquisition rows."""
  if not enabled:
    return data
  width = torch.as_tensor(magnitude, dtype=data.dtype, device=data.device)
  if width.ndim > 0 and width.numel() != data.shape[-1]:
    raise ValueError(
      f"Noise width has {width.numel()} channels for observation width "
      f"{data.shape[-1]}."
    )
  noise = (2.0 * torch.rand_like(data) - 1.0) * width
  mask = acquisition_env_mask(data.shape[0], acquisition_fraction, data.device).view(
    -1, *([1] * (data.ndim - 1))
  )
  return data + noise * mask


def role_noisy_projected_gravity(
  env: ManagerBasedRlEnv,
  acquisition_fraction: float | None,
  magnitude: float,
  enabled: bool = True,
) -> torch.Tensor:
  return _add_acquisition_uniform_noise(
    mj_mdp.projected_gravity(env), magnitude, acquisition_fraction, enabled
  )


def role_noisy_builtin_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  acquisition_fraction: float | None,
  magnitude: float,
  enabled: bool = True,
) -> torch.Tensor:
  return _add_acquisition_uniform_noise(
    mj_mdp.builtin_sensor(env, sensor_name), magnitude, acquisition_fraction, enabled
  )


def role_noisy_joint_pos_rel(
  env: ManagerBasedRlEnv,
  acquisition_fraction: float | None,
  magnitude: float,
  enabled: bool = True,
  biased: bool = True,
) -> torch.Tensor:
  return _add_acquisition_uniform_noise(
    mj_mdp.joint_pos_rel(env, biased=biased),
    magnitude,
    acquisition_fraction,
    enabled,
  )


def role_noisy_joint_vel_rel(
  env: ManagerBasedRlEnv,
  acquisition_fraction: float | None,
  magnitude: float,
  enabled: bool = True,
) -> torch.Tensor:
  return _add_acquisition_uniform_noise(
    mj_mdp.joint_vel_rel(env), magnitude, acquisition_fraction, enabled
  )


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


def role_noisy_motion_command_window(
  env: ManagerBasedRlEnv,
  command_name: str,
  acquisition_fraction: float | None,
  magnitude: tuple[float, ...],
  enabled: bool = True,
) -> torch.Tensor:
  """Command-window perturbation restricted to Stage-II acquisition rows."""
  clean = motion_command_window(env, command_name)
  return _add_acquisition_uniform_noise(clean, magnitude, acquisition_fraction, enabled)


def motion_command_token(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Current command token ``g_t = [v_ref, w_ref, g_ref, q_ref]``, shape ``(N, 38)``.

  RGMT Eq. 4 gives the critic ``s_t = [o_t, g_t, o_t^priv]`` -- the *single* current
  reference token, not the actor's 21-token window. The centre of the window is
  exactly ``g_t`` (offset 0 of ``window_offsets``).
  """
  command = _cmd(env, command_name)
  return command.command_window()[:, command.cfg.command_window_radius]


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
