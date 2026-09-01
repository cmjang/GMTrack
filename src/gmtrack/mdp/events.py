"""Event terms for Extreme-RGMT.

The paper applies its perturbation protocol to all Stage-I environments but only the
acquisition role in Stage II (Sec. IV-B2).  mjlab's built-in event functions operate
on the ``env_ids`` they receive, so the wrappers below intersect that set with the
PACE acquisition partition before delegating to the upstream implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import dr, push_by_setting_velocity
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from gmtrack.mdp.rewards import _cmd
from gmtrack.pace import pace_env_split

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def _acquisition_env_ids(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
) -> torch.Tensor:
  ids = (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None
    else env_ids.to(device=env.device, dtype=torch.long)
  )
  if acquisition_fraction is None:
    return ids
  split = pace_env_split(acquisition_fraction, env.num_envs)
  return ids[ids < split]


def role_push_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    push_by_setting_velocity(env, ids, velocity_range, asset_cfg)


@requires_model_fields("geom_friction")
def role_geom_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  ranges,
  asset_cfg: SceneEntityCfg,
  operation: str = "abs",
  shared_random: bool = False,
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.geom_friction(
      env,
      ids,
      ranges,
      asset_cfg,
      operation=operation,
      shared_random=shared_random,
    )


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def role_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  ranges,
  asset_cfg: SceneEntityCfg,
  operation: str = "scale",
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.body_mass(env, ids, ranges, asset_cfg, operation=operation)


@requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)
def role_body_com_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  ranges,
  asset_cfg: SceneEntityCfg,
  operation: str = "add",
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.body_com_offset(env, ids, ranges, asset_cfg, operation=operation)


@requires_model_fields("actuator_forcerange", "jnt_actfrcrange", "tendon_actfrcrange")
def role_effort_limits(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  effort_limit_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  operation: str = "scale",
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.effort_limits(env, ids, effort_limit_range, asset_cfg, operation=operation)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def role_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
  operation: str = "scale",
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.pd_gains(env, ids, kp_range, kd_range, asset_cfg, operation=operation)


def role_encoder_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  bias_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.encoder_bias(env, ids, bias_range, asset_cfg)


@requires_model_fields("dof_armature", recompute=RecomputeLevel.set_const_0)
def role_joint_armature(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  acquisition_fraction: float | None,
  ranges,
  asset_cfg: SceneEntityCfg,
  operation: str = "abs",
) -> None:
  ids = _acquisition_env_ids(env, env_ids, acquisition_fraction)
  if ids.numel() > 0:
    dr.joint_armature(env, ids, ranges, asset_cfg, operation=operation)


def recovery_assist_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Upward pulling force on the anchor body of in-window recovery environments.

  RGMT Sec. II-D: "we apply an upward pulling force with magnitude uniformly
  sampled from [0, 200]" N "to assist exploration at early training stages",
  linearly annealed over training. The per-episode magnitude (annealing included)
  is drawn by ``MultiMotionCommand._reset_recovery``; this term only routes it
  into the simulation.

  Runs with ``mode="step"``: ``xfrc_applied`` persists across environment resets
  (``Entity.reset`` does not clear it), so the wrench is rewritten for *every*
  environment each step -- zero outside the recovery window -- instead of only
  for the currently active ones.
  """
  del env_ids  # step mode fires unconditionally on all envs.
  command = _cmd(env, command_name)
  asset = env.scene[asset_cfg.name]
  force = torch.zeros(env.num_envs, 1, 3, device=env.device)
  active = command.in_recovery_window
  force[active, 0, 2] = command.recovery_assist_n[active]
  asset.write_external_wrench_to_sim(
    forces=force,
    torques=torch.zeros_like(force),
    body_ids=[command.robot_anchor_body_index],
  )
