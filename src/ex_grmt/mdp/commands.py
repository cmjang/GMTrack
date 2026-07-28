"""Multi-clip motion tracking command with PACE environment roles.

Derived from mjlab's single-clip ``MotionCommand``
(``mjlab/tasks/tracking/mdp/commands.py``, itself a re-implementation of
BeyondMimic / whole_body_tracking). The public property surface is kept
byte-compatible with mjlab's version so the vendored reward / termination /
observation terms work unchanged.

What is different:

* A :class:`~ex_grmt.mdp.motion_library.MotionLibrary` replaces the single
  ``MotionLoader``; every environment carries its own ``motion_id``.
* Adaptive difficulty sampling (Eq. 12-13) spans every (clip, bin) pair of a motion
  subset instead of the bins of one clip.
* Environments are split into PACE roles (paper Sec. V-A): a fraction ``xi`` of them
  are *acquisition* environments that sample the challenging set adaptively, the rest
  are *consolidation* environments that sample the mastered set uniformly.
* A local reference window ``g_{t-L:t+L}`` (21 tokens x 38 dims) is exposed for the
  command encoder, along with the STAR difficulty weight ``w_t``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from ex_grmt.mdp.motion_library import MotionLibrary
from ex_grmt.mdp.sampling import AdaptiveBinSampler

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Index of the floating-base body inside ``cfg.body_names``. mjlab's MotionCommand
# hard-codes the same assumption (``self.body_pos_w[env_ids, 0]`` is the root state
# written to the free joint), so the tracked-body tuple must start with the root link.
_ROOT = 0


class MultiMotionCommand(CommandTerm):
  """Reference-motion command over a library of clips."""

  cfg: MultiMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.lib = MotionLibrary.from_manifest(
      cfg.manifest,
      body_indexes=self.body_indexes,
      device=self.device,
      subset=_read_clip_names(cfg.clip_subset),
      bin_seconds=cfg.bin_seconds,
    )
    print(f"[ex-grmt] {self.lib}")

    # Root-body views into the packed library (no copy; used by the command window).
    self._root_quat = self.lib.body_quat_w[:, _ROOT]
    self._root_lin_vel = self.lib.body_lin_vel_w[:, _ROOT]
    self._root_ang_vel = self.lib.body_ang_vel_w[:, _ROOT]
    self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device)

    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    self._setup_roles()

    self.window_offsets = torch.arange(
      -cfg.command_window_radius,
      cfg.command_window_radius + 1,
      device=self.device,
      dtype=torch.long,
    )
    self.num_window_tokens = int(self.window_offsets.numel())
    self.command_token_dim = 9 + self.lib.joint_pos.shape[1]

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    for key in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
      "sampling_top1_prob",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  # -- PACE roles -----------------------------------------------------------

  def _setup_roles(self) -> None:
    """Partition environment indices into acquisition / consolidation groups."""
    cfg = self.cfg
    all_ids = torch.arange(self.num_envs, device=self.device)

    acq_names = _read_clip_names(cfg.acquisition_clips)
    con_names = _read_clip_names(cfg.consolidation_clips)

    if cfg.acquisition_fraction is None:
      # Stage I: one homogeneous group over the whole library.
      self.acq_env_ids = all_ids
      self.con_env_ids = all_ids[:0]
      acq_clip_ids = self._resolve_clip_ids(acq_names)
      con_clip_ids = None
    else:
      if not 0.0 < cfg.acquisition_fraction < 1.0:
        raise ValueError(
          f"acquisition_fraction must be in (0, 1), got {cfg.acquisition_fraction}."
        )
      if con_names is None:
        raise ValueError(
          "Stage II needs `consolidation_clips` (the mastered set D_m) alongside "
          "`acquisition_fraction`."
        )
      split = int(cfg.acquisition_fraction * self.num_envs)
      split = min(max(split, 1), self.num_envs - 1)
      self.acq_env_ids = all_ids[:split]
      self.con_env_ids = all_ids[split:]
      acq_clip_ids = self._resolve_clip_ids(acq_names)
      con_clip_ids = self._resolve_clip_ids(con_names)

    self.sampler_acq = self._make_sampler(acq_clip_ids)
    self.sampler_con = (
      self._make_sampler(con_clip_ids) if con_clip_ids is not None else None
    )

    print(
      f"[ex-grmt] PACE roles: acquisition={self.acq_env_ids.numel()} envs over "
      f"{self.sampler_acq.num_clips} clips / {self.sampler_acq.num_valid_bins} bins; "
      f"consolidation={self.con_env_ids.numel()} envs over "
      f"{self.sampler_con.num_clips if self.sampler_con else 0} clips"
    )

  def _resolve_clip_ids(self, names: list[str] | None) -> torch.Tensor:
    if names is None:
      return torch.arange(self.lib.num_clips, dtype=torch.long, device=self.device)
    return self.lib.clip_ids_by_name(names)

  def _make_sampler(self, clip_ids: torch.Tensor) -> AdaptiveBinSampler:
    return AdaptiveBinSampler(
      clip_ids=clip_ids,
      clip_bins=self.lib.clip_bins[clip_ids],
      max_bins=self.lib.max_bins,
      num_library_clips=self.lib.num_clips,
      kernel_size=self.cfg.adaptive_kernel_size,
      kernel_lambda=self.cfg.adaptive_lambda,
      uniform_ratio=self.cfg.adaptive_uniform_ratio,
      alpha=self.cfg.adaptive_alpha,
      max_count=self.cfg.adaptive_max_count,
      device=self.device,
    )

  # -- reference lookups ----------------------------------------------------

  @property
  def frames(self) -> torch.Tensor:
    """``(N,)`` global row of each environment's current reference frame."""
    return self.lib.frame_index(self.motion_ids, self.time_steps)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.lib.joint_pos[self.frames]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.lib.joint_vel[self.frames]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.lib.body_pos_w[self.frames] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.lib.body_quat_w[self.frames]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.lib.body_lin_vel_w[self.frames]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.lib.body_ang_vel_w[self.frames]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.lib.body_pos_w[self.frames, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.lib.body_quat_w[self.frames, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.lib.body_lin_vel_w[self.frames, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.lib.body_ang_vel_w[self.frames, self.motion_anchor_body_index]

  # -- robot state (mirrors mjlab's naming) ---------------------------------

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  # -- Extreme-RGMT specific outputs ----------------------------------------

  def command_window(self) -> torch.Tensor:
    """Local reference window ``g_{t-L:t+L}``, shape ``(N, 2L+1, 9 + J)``.

    Each token is ``[v_ref, w_ref, g_ref, q_ref]`` (paper Eq. 2). Linear and angular
    velocities and the gravity direction are expressed **in the reference root frame
    at that token's own timestamp**, which makes every token a self-contained
    egocentric description of the reference and keeps the window invariant to where
    the clip happens to sit in world coordinates.

    Frames outside the clip are clamped to its first/last frame rather than wrapped,
    so the window never bleeds into a neighbouring clip.
    """
    idx = self.lib.window_index(self.motion_ids, self.time_steps, self.window_offsets)
    n, w = idx.shape
    flat = idx.reshape(-1)

    quat = self._root_quat[flat]
    q_inv = quat_inv(quat)
    v_b = quat_apply(q_inv, self._root_lin_vel[flat])
    w_b = quat_apply(q_inv, self._root_ang_vel[flat])
    g_b = quat_apply(q_inv, self._gravity_w.expand(flat.shape[0], 3))
    q_ref = self.lib.joint_pos[flat]

    return torch.cat([v_b, w_b, g_b, q_ref], dim=-1).view(n, w, -1)

  def star_meta(self) -> torch.Tensor:
    """``(N, 2)`` of ``[difficulty weight w_t, flat bin id]`` for STAR.

    ``w_t = B * p_{b_t}`` (Eq. 20). Consolidation environments get weight 0: their
    clips are not in the acquisition sampler's subset, so they fall into STAR's
    low-difficulty group ``E`` and are never picked for fragment resampling (STAR
    only acts on the acquisition side).
    """
    bins = self.lib.bin_of(self.motion_ids, self.time_steps)
    weight = self.sampler_acq.bin_weight(self.motion_ids, bins)
    bin_id = self.sampler_acq.flat_bin_id(self.motion_ids, bins)
    return torch.stack([weight, bin_id.to(weight.dtype)], dim=-1)

  # -- CommandTerm hooks ----------------------------------------------------

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._record_failures(env_ids)
    self._sample_reference_states(env_ids)

    root_pos = self.body_pos_w[env_ids, _ROOT].clone()
    root_ori = self.body_quat_w[env_ids, _ROOT].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, _ROOT].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, _ROOT].clone()

    # Reference-state-initialisation noise (paper Table II command perturbations).
    ranges = torch.tensor(
      [
        self.cfg.pose_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += samples[:, 0:3]
    root_ori = quat_mul(
      quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]), root_ori
    )

    ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += samples[:, :3]
    root_ang_vel += samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )

    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  def _record_failures(self, env_ids: torch.Tensor) -> None:
    """Attribute terminations (not time-outs) to the bin they happened in."""
    if self.cfg.sampling_mode != "adaptive":
      return
    terminated = self._env.termination_manager.terminated[env_ids]
    if not bool(terminated.any()):
      return
    failed = env_ids[terminated]
    bins = self.lib.bin_of(self.motion_ids[failed], self.time_steps[failed])
    self.sampler_acq.record_failures(self.motion_ids[failed], bins)
    if self.sampler_con is not None:
      self.sampler_con.record_failures(self.motion_ids[failed], bins)

  def _sample_reference_states(self, env_ids: torch.Tensor) -> None:
    """Choose ``(motion_id, time_step)`` per resetting environment, by role."""
    if self.cfg.sampling_mode == "start":
      # Deterministic replay: keep the clip, rewind to frame 0.
      self.time_steps[env_ids] = 0
      return

    is_con = torch.isin(env_ids, self.con_env_ids)

    for ids, sampler, uniform in (
      (env_ids[~is_con], self.sampler_acq, self.cfg.sampling_mode == "uniform"),
      (env_ids[is_con], self.sampler_con, True),
    ):
      if ids.numel() == 0 or sampler is None:
        continue
      clips, bins = (
        sampler.sample_uniform(int(ids.numel()))
        if uniform
        else sampler.sample(int(ids.numel()))
      )
      jitter = torch.randint(
        0, self.lib.frames_per_bin, (int(ids.numel()),), device=self.device
      )
      local = bins * self.lib.frames_per_bin + jitter
      self.motion_ids[ids] = clips
      self.time_steps[ids] = torch.clamp(
        local, torch.zeros_like(local), self.lib.clip_len[clips] - 1
      )

    h, top1 = self.sampler_acq.entropy_stats()
    self.metrics["sampling_entropy"][:] = h
    self.metrics["sampling_top1_prob"][:] = top1

  def set_clip(
    self,
    env_ids: torch.Tensor,
    motion_ids: torch.Tensor,
    frames: torch.Tensor | int = 0,
  ) -> None:
    """Pin environments to specific clips/frames and snap the robot onto the reference.

    Deterministic -- no RSI perturbation. Used by evaluation and stratification, which
    need a controlled starting condition per clip rather than the training sampler's
    difficulty-weighted draw.
    """
    if isinstance(frames, int):
      frames = torch.full_like(env_ids, frames)
    self.motion_ids[env_ids] = motion_ids
    self.time_steps[env_ids] = frames
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, _ROOT],
      self.body_quat_w[env_ids, _ROOT],
      self.body_lin_vel_w[env_ids, _ROOT],
      self.body_ang_vel_w[env_ids, _ROOT],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def update_relative_body_poses(self) -> None:
    """Express the reference bodies in the robot's current anchor frame.

    Copied from mjlab's ``MotionCommand`` so termination checks compare like with
    like: the reference is re-anchored to the robot's horizontal position and yaw,
    keeping only the height and full orientation of the reference.
    """
    num_bodies = len(self.cfg.body_names)
    anchor_pos_w = self.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat_w = self.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_pos_w = self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_w = self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    delta_pos_w = robot_anchor_pos_w.clone()
    delta_pos_w[..., 2] = anchor_pos_w[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w
    )

  def _update_command(self) -> None:
    self.time_steps += 1
    done = torch.where(self.time_steps >= self.lib.clip_len[self.motion_ids])[0]
    if done.numel() > 0:
      self._resample_command(done)
      # _resample_command writes qpos/qvel but does not refresh derived quantities;
      # forward() so update_relative_body_poses reads the post-teleport anchor.
      self._env.sim.forward()

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      self.sampler_acq.step_ema()
      if self.sampler_con is not None:
        self.sampler_con.step_ema()


def _read_clip_names(spec: str | Path | list[str] | None) -> list[str] | None:
  """Accept a name list, a JSON file of names, or None (= use everything)."""
  if spec is None:
    return None
  if isinstance(spec, list):
    return spec
  with Path(spec).open() as f:
    payload = json.load(f)
  if isinstance(payload, dict):
    if "clips" in payload:
      return [c["name"] if isinstance(c, dict) else c for c in payload["clips"]]
    if "names" in payload:
      return list(payload["names"])
    raise ValueError(f"{spec}: expected a 'clips' or 'names' key.")
  return [c["name"] if isinstance(c, dict) else c for c in payload]


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  """Configuration for :class:`MultiMotionCommand`."""

  manifest: str
  """Path to the clip manifest produced by ``ex_grmt.scripts.prepare_motions``."""
  anchor_body_name: str
  body_names: tuple[str, ...]
  """Tracked bodies. **Must start with the floating-base root link.**"""
  entity_name: str

  clip_subset: str | list[str] | None = None
  """Restrict the loaded library. None loads every clip in the manifest."""
  acquisition_clips: str | list[str] | None = None
  """Challenging set ``D_c``. None (Stage I) means "the whole library"."""
  consolidation_clips: str | list[str] | None = None
  """Mastered set ``D_m``. Required when ``acquisition_fraction`` is set."""
  acquisition_fraction: float | None = None
  """``xi`` in Algorithm 1. None disables the PACE split (Stage I). Paper uses 0.8."""

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.1, 0.1)

  bin_seconds: float = 1.0
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  adaptive_max_count: float = 1.0
  """``c_max`` in Eq. (13). mjlab omits this clip; the paper specifies it."""
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  command_window_radius: int = 10
  """``L``. The window has ``2L + 1`` tokens; the paper uses 21, i.e. L = 10."""

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)
