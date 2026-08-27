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
  command encoder, along with the STAR difficulty weight ``w_t``. An opt-in
  SONIC-style closed-loop heading variant appends six root-orientation channels to
  each token without changing that 38D baseline.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from ex_grmt.mdp.motion_library import MotionLibrary
from ex_grmt.mdp.sampling import AdaptiveBinSampler
from ex_grmt.pace import pace_env_split

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Index of the floating-base body inside ``cfg.body_names``. mjlab's MotionCommand
# hard-codes the same assumption (``self.body_pos_w[env_ids, 0]`` is the root state
# written to the free joint), so the tracked-body tuple must start with the root link.
_ROOT = 0


def _clamp_training_start_frame(
  local_frames: torch.Tensor, clip_lengths: torch.Tensor
) -> torch.Tensor:
  """Clamp sampled starts before the terminal reference frame.

  The terminal frame is a target state, not a valid episode start: starting there
  makes ``motion_sequence_end`` fire before the policy executes a transition.
  """
  last_start = torch.clamp_min(clip_lengths - 2, 0)
  return torch.clamp_min(torch.minimum(local_frames, last_start), 0)


def relative_root_orientation_6d(
  robot_root_quat_w: torch.Tensor, ref_root_quat_w: torch.Tensor
) -> torch.Tensor:
  """SONIC-style reference root orientation relative to the current robot root.

  The inputs must be broadcast-compatible ``(..., 4)`` scalar-first quaternions.
  The returned ``(..., 6)`` representation is formed from
  ``q_robot_current^-1 * q_ref_future`` and flattens the first two matrix columns in
  row-major order. This deliberately matches SONIC's released
  ``matrix[..., :2].reshape(...)`` channel order exactly.

  Kept independent of the command/environment classes so the quaternion convention
  and channel ordering can be unit-tested without constructing a simulator.
  """
  relative_quat = quat_mul(quat_inv(robot_root_quat_w), ref_root_quat_w)
  matrix = matrix_from_quat(relative_quat)
  return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


class MultiMotionCommand(CommandTerm):
  """Reference-motion command over a library of clips."""

  cfg: MultiMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    actor_offsets = _parse_window_offsets(cfg)
    critic_offsets = _parse_critic_window_offsets(cfg)
    reconstruction_offsets = _parse_reconstruction_window_offsets(cfg)

    if cfg.require_v1_stratification:
      if (
        cfg.stratification_challenging_manifest is None
        or cfg.stratification_mastered_manifest is None
      ):
        raise ValueError(
          "Strict v1 tasks require validation manifests for both challenging D_c "
          "and mastered D_m."
        )
      from ex_grmt.protocol import validate_stage2_manifests

      self.stratification_report = validate_stage2_manifests(
        cfg.manifest,
        cfg.stratification_mastered_manifest,
        cfg.stratification_challenging_manifest,
      )
    else:
      self.stratification_report = None

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

    if not 0.0 <= cfg.recovery_probability <= 1.0:
      raise ValueError("recovery_probability must be in [0, 1].")
    if cfg.recovery_probability > 0.0:
      # ``make_ex_grmt_env_cfg`` only *builds* the assistance-force event term when it
      # is called with recovery enabled, so raising the probability on an already-built
      # config -- which is all a `--env.commands.motion.recovery-probability` override
      # can do -- gives fallen resets and a termination shield with no upward force at
      # all. That silently removes the one mechanism RGMT Sec. II-D provides for
      # escaping fallen states. Use the ExGRMT-Stage1-Recovery-* task instead.
      if "recovery_assist" not in env.cfg.events:
        raise ValueError(
          "recovery_probability > 0 but the environment has no 'recovery_assist' "
          "event term: the assistance force can only be built by calling "
          "make_ex_grmt_env_cfg(recovery_probability=...), not by overriding the "
          "command afterwards. Use the ExGRMT-Stage1-Recovery-Flat-Unitree-G1 task."
        )
      root_low, root_high = cfg.recovery_root_height_range
      if root_low < 0.0 or root_low > root_high:
        raise ValueError("recovery_root_height_range must be non-negative and ordered.")
      tilt_low, tilt_high = cfg.recovery_root_tilt_range
      if tilt_low < 0.0 or tilt_low > tilt_high or tilt_high > math.pi:
        raise ValueError("recovery_root_tilt_range must be ordered within [0, pi].")
      joint_low, joint_high = cfg.recovery_joint_position_jitter
      if joint_low > joint_high:
        raise ValueError("recovery_joint_position_jitter must be ordered.")
      print(
        "[ex-grmt] RGMT random-pose recovery: "
        f"p={cfg.recovery_probability:g}, root_z={cfg.recovery_root_height_range}, "
        f"tilt={cfg.recovery_root_tilt_range}; no recovery-motion data"
      )

    # Root-body views into the packed library (no copy; used by the command window).
    self._root_quat = self.lib.body_quat_w[:, _ROOT]
    self._root_lin_vel = self.lib.body_lin_vel_w[:, _ROOT]
    self._root_ang_vel = self.lib.body_ang_vel_w[:, _ROOT]
    self._gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device)

    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # Fall recovery (RGMT, arXiv:2601.23080v1, Sec. II-D). ``recovery_mask`` marks
    # environments whose current episode started from a randomized fallen pose;
    # ``recovery_assist_raw_n`` holds that episode's U[0, 200] N draw *before* the
    # training-progress anneal, which ``recovery_assist_n`` applies at read time.
    # Splitting them keeps the applied force a pure function of the current clock:
    # the environment is reset once during ``RslRlVecEnvWrapper.__init__``, which is
    # before ``runner.load()`` restores the clock, so an episode-time draw would hand
    # the first post-resume episodes a force annealed against a clock of zero.
    self.recovery_mask = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.recovery_assist_raw_n = torch.zeros(self.num_envs, device=self.device)
    self._recovery_window_steps = max(
      1, int(round(cfg.recovery_window_s / env.step_dt))
    )
    # Environment steps taken *with fall recovery enabled*, which is the clock the
    # assistance anneal runs on ("linearly annealed over training iterations", RGMT
    # Sec. II-D). It deliberately is not mjlab's ``common_step_counter``: that counts
    # the whole training history, so enabling recovery on a resume from a run that
    # never had it starts the anneal already exhausted and the assistance force is
    # identically zero for every recovery episode. Checkpointed by
    # ``ExGRMTOnPolicyRunner`` so the anneal survives a restart of a recovery run.
    self.recovery_steps_elapsed = 0
    self._recovery_anneal_steps = max(1, int(cfg.recovery_assist_anneal_steps))

    self._setup_roles()

    self.window_offsets = torch.tensor(
      actor_offsets, device=self.device, dtype=torch.long
    )
    self.critic_window_offsets = torch.tensor(
      critic_offsets, device=self.device, dtype=torch.long
    )
    self.reconstruction_window_offsets = torch.tensor(
      reconstruction_offsets, device=self.device, dtype=torch.long
    )
    self.num_window_tokens = int(self.window_offsets.numel())
    self.command_token_dim = (
      9 + self.lib.joint_pos.shape[1] + (6 if cfg.heading_closed_loop else 0)
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Ghost model for reference-motion visualization (green semi-transparent robot).
    self._ghost_model = None
    self._ghost_color = np.array([0.5, 0.7, 0.5, 0.5], dtype=np.float32)

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
      if con_names is None:
        raise ValueError(
          "Stage II needs `consolidation_clips` (the mastered set D_m) alongside "
          "`acquisition_fraction`."
        )
      # Shared with StarRolloutStorage: the storage derives an environment's role
      # from its index alone, so both sides must round identically.
      split = pace_env_split(cfg.acquisition_fraction, self.num_envs)
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
      max_count_over_mean=self.cfg.adaptive_max_count_over_mean,
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

  def _tokens_at(self, idx: torch.Tensor) -> torch.Tensor:
    """Build reference tokens at packed-library frame indices.

    This is shared by the actor and privileged-critic windows so their reference
    frames use exactly the same egocentric convention.
    """
    n, w = idx.shape
    flat = idx.reshape(-1)

    quat = self._root_quat[flat]
    q_inv = quat_inv(quat)
    v_b = quat_apply(q_inv, self._root_lin_vel[flat])
    w_b = quat_apply(q_inv, self._root_ang_vel[flat])
    g_b = quat_apply(q_inv, self._gravity_w.expand(flat.shape[0], 3))
    q_ref = self.lib.joint_pos[flat]

    token_parts = [v_b, w_b, g_b, q_ref]
    if self.cfg.heading_closed_loop:
      robot_root_quat = (
        self.robot_body_quat_w[:, _ROOT, None, :].expand(-1, w, -1).reshape(-1, 4)
      )
      token_parts.append(relative_root_orientation_6d(robot_root_quat, quat))

    return torch.cat(token_parts, dim=-1).view(n, w, self.command_token_dim)

  def command_window(self) -> torch.Tensor:
    """Actor reference window, shape (N, W_actor, 9 + J [+ 6]).

    Each token is ``[v_ref, w_ref, g_ref, q_ref]`` (paper Eq. 2). Linear and angular
    velocities and the gravity direction are expressed **in the reference root frame
    at that token's own timestamp**, which makes every token a self-contained
    egocentric description of the reference and keeps the window invariant to where
    the clip happens to sit in world coordinates.

    Logical Stage-II fragments are views into their complete parent sequence, so
    offsets may read genuine context outside the sampleable fragment. Only offsets
    beyond the physical parent sequence are clamped; :meth:`command_window_valid_mask`
    identifies those padded tokens.

    With ``heading_closed_loop=True``, each token additionally ends in the SONIC-style
    six-dimensional relative root orientation
    ``R(q_robot_current^-1 * q_ref_token)[:, :2]``. The robot pelvis orientation is
    current feedback shared by every token; the reference orientation comes from
    that token's (possibly boundary-clamped) timestamp.

    Explicit offsets support a near-dense/far-sparse causal layout inspired by DAJI
    (arXiv:2605.14417) and TeleGate (arXiv:2602.09628). The default radius still
    resolves to the dense symmetric window used by existing tasks.
    """
    idx = self.lib.window_index(self.motion_ids, self.time_steps, self.window_offsets)
    return self._tokens_at(idx)

  def command_window_valid_mask(self) -> torch.Tensor:
    """Return one validity bit per actor command token, shape (N, W_actor).

    The mask is relative to the physical parent sequence, not a Stage-II logical
    fragment. It therefore marks only true sequence-end padding and reveals neither
    the fragment boundary nor a time-to-reset countdown. At a physical sequence
    start it also gives the actor an explicit cold-start signal instead of pretending
    repeated first frames are executed history.
    """
    _, valid = self.lib.window_index_and_mask(
      self.motion_ids, self.time_steps, self.window_offsets
    )
    return valid

  def critic_command_window(self) -> torch.Tensor:
    """Privileged future reference tokens, shape ``(N, W_critic, C)``.

    The asymmetric critic receives exact positive offsets during training only. Its
    history inputs remain alongside this window so the value is ``V(h, s, g_future)``
    rather than a state-only critic for a history-dependent policy, following
    Baisero & Amato (arXiv:2105.11674) and Informed Asymmetric Actor-Critic
    (arXiv:2509.26000). GigaBrain-WBC-0.5 (arXiv:2608.18234) likewise combines
    multi-frame future reference with proprioceptive/action history for its critic.
    """
    if self.critic_window_offsets.numel() == 0:
      raise ValueError(
        "critic_command_window requires non-empty critic_window_offsets."
      )
    idx = self.lib.window_index(
      self.motion_ids, self.time_steps, self.critic_window_offsets
    )
    return self._tokens_at(idx)

  def critic_command_window_valid_mask(self) -> torch.Tensor:
    """Validity bits for privileged future tokens at physical sequence ends."""
    if self.critic_window_offsets.numel() == 0:
      raise ValueError(
        "critic_command_window_valid_mask requires critic_window_offsets."
      )
    _, valid = self.lib.window_index_and_mask(
      self.motion_ids, self.time_steps, self.critic_window_offsets
    )
    return valid

  def reconstruction_command_window(self) -> torch.Tensor:
    """Training target at sparse future offsets, shape ``(N, W_recon, C)``.

    TeleGate (arXiv:2602.09628) uses the ``(+5, +10, +20)`` prediction layout.
    This target is consumed only by the causal actor's auxiliary reconstruction loss
    and is not an actor observation or an exported deployment input.
    """
    if self.reconstruction_window_offsets.numel() == 0:
      raise ValueError(
        "reconstruction_command_window requires reconstruction_window_offsets."
      )
    idx = self.lib.window_index(
      self.motion_ids, self.time_steps, self.reconstruction_window_offsets
    )
    return self._tokens_at(idx)

  def reconstruction_command_window_valid_mask(self) -> torch.Tensor:
    """Validity bits for sparse auxiliary targets at parent-sequence ends."""
    if self.reconstruction_window_offsets.numel() == 0:
      raise ValueError(
        "reconstruction_command_window_valid_mask requires reconstruction offsets."
      )
    _, valid = self.lib.window_index_and_mask(
      self.motion_ids, self.time_steps, self.reconstruction_window_offsets
    )
    return valid

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

  # -- fall recovery (RGMT Sec. II-D) ---------------------------------------

  @property
  def recovery_assist_anneal(self) -> float:
    """Linear [1, 0] training-progress factor on the assistance force (RGMT II-D)."""
    return max(0.0, 1.0 - self.recovery_steps_elapsed / self._recovery_anneal_steps)

  @property
  def recovery_assist_n(self) -> torch.Tensor:
    """``(N,)`` upward assistance in newtons, annealed at the current clock.

    Evaluated on read rather than frozen at episode reset so the value can never be
    stale with respect to ``recovery_steps_elapsed``. Over one 10 s episode the
    anneal moves by 500/2400000 = 0.021%, so this is indistinguishable from RGMT's
    per-episode magnitude, and it removes the reset-before-checkpoint-load hazard.
    """
    return self.recovery_assist_anneal * self.recovery_assist_raw_n

  def compute(self, dt: float) -> None:
    """Advance the anneal clock once per *environment step*, then defer to mjlab.

    ``ManagerBasedRlEnv.reset`` also calls ``command_manager.compute``, with
    ``dt=0.0``; only ``step`` passes the real ``step_dt``. Counting inside
    ``_update_command`` would therefore add one tick per reset on top of the steps.
    """
    if dt > 0.0 and self.cfg.recovery_probability > 0.0:
      self.recovery_steps_elapsed += 1
    super().compute(dt)

  @property
  def in_recovery_window(self) -> torch.Tensor:
    """``(N,)`` bool: recovery environments still inside their 3 s window.

    While True, the instability terminations are suspended (see the
    ``*_outside_recovery`` terms) so the policy can complete stand-up and
    re-stabilization within the same episode; once the window elapses the checks
    re-engage, which is RGMT's "fails to recover within this window -> terminate".
    """
    return self.recovery_mask & (
      self._env.episode_length_buf < self._recovery_window_steps
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # super().reset() runs failure attribution for the episode that just ended, so
    # it must see the OLD recovery mask; the new draw happens after.
    extras = super().reset(env_ids)
    assert isinstance(env_ids, torch.Tensor)
    self._reset_recovery(env_ids)
    return extras

  def _reset_recovery(self, env_ids: torch.Tensor) -> None:
    """Reset selected acquisition environments to randomized unstable poses.

    RGMT Sec. II-D specifies the 15% draw, randomized poses, upward assistance and
    three-second shield, but does not publish the pose distribution. This
    implementation constructs the pose at runtime: a low root, a broad non-upright
    orientation, and bounded jitter around the already sampled ordinary reference
    joint pose. The ordinary ``motion_ids`` and ``time_steps`` are deliberately left
    untouched, so no fall/get-up demonstration or recovery-motion dataset is used.
    """
    cfg = self.cfg
    self.recovery_mask[env_ids] = False
    self.recovery_assist_raw_n[env_ids] = 0.0
    if cfg.recovery_probability <= 0.0:
      return

    draw = torch.rand(env_ids.shape[0], device=self.device) < cfg.recovery_probability
    # Recovery belongs to the training perturbation protocol, which Extreme-RGMT
    # applies "during Stage I and in the acquisition environments of Stage II"
    # (Sec. IV-B2). Consolidation environments must stay clean: their rollouts feed
    # the pi_ref alignment loss (Eq. 15), and pi_ref never saw fallen states.
    draw &= torch.isin(env_ids, self.acq_env_ids)
    ids = env_ids[draw]
    if ids.numel() == 0:
      return
    self.recovery_mask[ids] = True

    # A recovery reset must receive the full three-second shield. Ordinary training
    # starts may otherwise land in the last seconds of a clip, where
    # ``motion_sequence_end`` would reset the environment before recovery can finish.
    # Keep the same ordinary motion and move only an over-late reference timestamp to
    # the latest start that still contains the complete window.
    clip_lengths = self.lib.clip_len[self.motion_ids[ids]]
    latest_start = torch.clamp_min(clip_lengths - self._recovery_window_steps - 1, 0)
    self.time_steps[ids] = torch.minimum(self.time_steps[ids], latest_start)

    # Upward assistance magnitude ~ U[0, 200] N (RGMT Sec. II-D), one draw per
    # recovery episode. The linear anneal is applied by ``recovery_assist_n`` on
    # read, against the recovery-local clock rather than the global step counter.
    self.recovery_assist_raw_n[ids] = sample_uniform(
      cfg.recovery_assist_force_range[0],
      cfg.recovery_assist_force_range[1],
      (ids.numel(),),
      device=self.device,
    )

    root_pos = self._env.scene.env_origins[ids].clone()
    root_pos[:, 2] += sample_uniform(
      cfg.recovery_root_height_range[0],
      cfg.recovery_root_height_range[1],
      (ids.numel(),),
      device=self.device,
    )

    # Sample a tilt about a random horizontal axis, then a world-yaw rotation. The
    # lower bound keeps every selected pose materially non-upright. Scalar-first
    # quaternions match mjlab's convention.
    tilt = sample_uniform(
      cfg.recovery_root_tilt_range[0],
      cfg.recovery_root_tilt_range[1],
      (ids.numel(),),
      device=self.device,
    )
    axis_azimuth = sample_uniform(-math.pi, math.pi, (ids.numel(),), device=self.device)
    half_tilt = 0.5 * tilt
    root_tilt = torch.stack(
      [
        torch.cos(half_tilt),
        torch.cos(axis_azimuth) * torch.sin(half_tilt),
        torch.sin(axis_azimuth) * torch.sin(half_tilt),
        torch.zeros_like(half_tilt),
      ],
      dim=-1,
    )
    yaw = sample_uniform(-math.pi, math.pi, (ids.numel(),), device=self.device)
    zeros = torch.zeros_like(yaw)
    root_ori = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), root_tilt)

    frames = self.lib.frame_index(self.motion_ids[ids], self.time_steps[ids])
    joint_pos = self.lib.joint_pos[frames].clone()
    joint_pos += sample_uniform(
      cfg.recovery_joint_position_jitter[0],
      cfg.recovery_joint_position_jitter[1],
      joint_pos.shape,
      device=self.device,
    )
    self._write_reference_state_to_sim(
      ids,
      root_pos,
      root_ori,
      torch.zeros_like(root_pos),
      torch.zeros_like(root_pos),
      joint_pos,
      torch.zeros_like(joint_pos),
    )

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
    is_acquisition = torch.isin(env_ids, self.acq_env_ids)

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
    samples[~is_acquisition] = 0.0
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
    samples[~is_acquisition] = 0.0
    root_lin_vel += samples[:, :3]
    root_ang_vel += samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    joint_jitter = sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )
    joint_jitter[~is_acquisition] = 0.0
    joint_pos += joint_jitter

    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  def _record_failures(self, env_ids: torch.Tensor) -> None:
    """Attribute terminations (not time-outs) to the bin they happened in."""
    if self.cfg.sampling_mode != "adaptive":
      return
    terminated = self._env.termination_manager.terminated[env_ids]
    # Recovery episodes (RGMT Sec. II-D) fail because they start from a random
    # fallen state, not because the reference bin is hard to track. Counting them
    # would corrupt the failure statistic c_i behind Eq. (12).
    terminated = terminated & ~self.recovery_mask[env_ids]
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
      self.time_steps[ids] = _clamp_training_start_frame(
        local, self.lib.clip_len[clips]
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

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw a green semi-transparent ghost robot at the reference pose."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self._ghost_model is None:
      # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
      # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      for gi in range(self._ghost_model.ngeom):
        if (
          self._ghost_model.geom_contype[gi] != 0
          or self._ghost_model.geom_conaffinity[gi] != 0
        ):
          self._ghost_model.geom_rgba[gi, 3] = 0
        else:
          self._ghost_model.geom_rgba[gi] = self._ghost_color

    indexing = self.robot.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()

    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
      qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

      visualizer.add_ghost_mesh(
        qpos,
        model=self._ghost_model,
        label=f"ghost_{batch}",
      )

  def create_gui(
    self,
    name: str,
    server: Any,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Add a clip selector dropdown and frame scrubber to the viser GUI."""
    # Build clip name list with index labels.
    clip_names = self.lib.clip_names  # list[str] of length num_clips
    clip_options = [f"[{i:03d}] {n}" for i, n in enumerate(clip_names)]

    with server.gui.add_folder(name.capitalize()):
      clip_dropdown = server.gui.add_dropdown(
        "Clip",
        options=clip_options,
        initial_value=clip_options[0],
      )
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=500,
        step=1,
        initial_value=0,
      )

      @clip_dropdown.on_update
      def _(_) -> None:
        idx = get_env_idx()
        new_clip = clip_options.index(clip_dropdown.value)
        self.motion_ids[idx] = new_clip
        self.time_steps[idx] = 0
        # Update scrubber max to match the new clip length.
        scrubber.max = int(self.lib.clip_len[new_clip].item()) - 1
        scrubber.value = 0
        env_ids = torch.tensor([idx], device=self.device)
        self._write_reference_state_to_sim(
          env_ids,
          self.body_pos_w[env_ids, _ROOT],
          self.body_quat_w[env_ids, _ROOT],
          self.body_lin_vel_w[env_ids, _ROOT],
          self.body_ang_vel_w[env_ids, _ROOT],
          self.joint_pos[env_ids],
          self.joint_vel[env_ids],
        )
        if on_change is not None:
          on_change()

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = int(scrubber.value)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn, clip_dropdown)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def _update_command(self) -> None:
    self.time_steps += 1
    if self.cfg.clamp_at_end:
      last_frames = self.lib.clip_len[self.motion_ids] - 1
      self.time_steps = torch.minimum(self.time_steps, last_frames)
      self.update_relative_body_poses()
      return

    done = torch.where(self.time_steps >= self.lib.clip_len[self.motion_ids])[0]
    if done.numel() > 0:
      self._resample_command(done)
      # The mid-episode clip hop above snapped the robot onto a fresh reference; it
      # is no longer in a recovery scenario and must not keep its termination shield.
      self.recovery_mask[done] = False
      self.recovery_assist_raw_n[done] = 0.0
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


def _parse_window_offsets(
  cfg: MultiMotionCommandCfg,
) -> tuple[int, ...]:
  """Validate and resolve actor command-window offsets.

  Explicit actor offsets override command_window_radius. Invalid layouts fail at
  command construction instead of being sorted, clamped, or coerced, because a
  silent correction could leak future frames into an online actor.
  """
  offsets = (
    tuple(range(-cfg.command_window_radius, cfg.command_window_radius + 1))
    if cfg.command_window_offsets is None
    else cfg.command_window_offsets
  )
  if any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets):
    raise TypeError(
      f"command_window_offsets must contain only integer offsets, got {offsets}."
    )
  if any(
    left >= right for left, right in zip(offsets, offsets[1:], strict=False)
  ):
    raise ValueError(
      f"command_window_offsets must be strictly increasing, got {offsets}."
    )
  if 0 not in offsets:
    raise ValueError(
      "command_window_offsets must contain offset 0 for the current reference frame, "
      f"got {offsets}."
    )
  if cfg.require_causal_window and any(offset > 0 for offset in offsets):
    raise ValueError(
      "require_causal_window=True forbids future actor references, got "
      f"command_window_offsets={offsets}."
    )
  return offsets


def _parse_positive_window_offsets(
  offsets: tuple[int, ...] | None, field_name: str
) -> tuple[int, ...]:
  """Validate a strictly increasing, future-only optional window."""
  if offsets is None:
    return ()
  if not offsets:
    raise ValueError(f"{field_name} must be non-empty when configured.")
  if any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets):
    raise TypeError(f"{field_name} must contain only integer offsets, got {offsets}.")
  if any(offset <= 0 for offset in offsets):
    raise ValueError(f"{field_name} must contain only positive offsets, got {offsets}.")
  if any(
    left >= right for left, right in zip(offsets, offsets[1:], strict=False)
  ):
    raise ValueError(f"{field_name} must be strictly increasing, got {offsets}.")
  return offsets


def _parse_critic_window_offsets(
  cfg: MultiMotionCommandCfg,
) -> tuple[int, ...]:
  """Resolve privileged-critic future offsets without sorting or coercion."""
  return _parse_positive_window_offsets(
    cfg.critic_window_offsets, "critic_window_offsets"
  )


def _parse_reconstruction_window_offsets(
  cfg: MultiMotionCommandCfg,
) -> tuple[int, ...]:
  """Resolve auxiliary future-prediction offsets without silent correction."""
  return _parse_positive_window_offsets(
    cfg.reconstruction_window_offsets, "reconstruction_window_offsets"
  )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  """Configuration for :class:`MultiMotionCommand`."""

  manifest: str
  """Path to a complete-sequence or post-Stage-I logical-clip manifest."""
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
  require_v1_stratification: bool = False
  """Fail closed unless Stage-II manifests carry valid v1 stratification provenance."""
  stratification_mastered_manifest: str | None = None
  """Authenticated D_m input used only for strict protocol validation."""
  stratification_challenging_manifest: str | None = None
  """Authenticated D_c input used only for strict protocol validation."""

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (0.0, 0.0)

  bin_seconds: float = 1.0
  adaptive_kernel_size: int = 1
  """Faithful Eq. (12)-(13) default. Values above 1 enable an unpublished smoothing
  ablation and must not be used for the main reproduction."""
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  adaptive_max_count_over_mean: float = 200.0
  """``c_max`` multiplier in Eq. (13). Following SONIC's public training release
  (``adp_samp_failure_rate_max_over_mean``), the actual clip bound is
  ``mean(active failed EMA) * adaptive_max_count_over_mean``.
  SONIC releases use 200; the library default is 50. Value is only meaningful
  when ``sampling_mode`` is ``"adaptive"``."""
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  clamp_at_end: bool = False
  """Hold the last reference frame instead of resampling a new clip.

  This is an evaluation-only control: deterministic full-sequence rollouts must score
  the final transition without teleporting the robot back to frame zero.
  """

  command_window_radius: int = 10
  """``L``. The window has ``2L + 1`` tokens; the paper uses 21, i.e. L = 10."""
  command_window_offsets: tuple[int, ...] | None = None
  """Explicit actor-window offsets, overriding command_window_radius.

  The causal task uses a configurable near-dense/far-sparse layout inspired by DAJI
  (arXiv:2605.14417) and TeleGate (arXiv:2602.09628). Offsets must be strictly
  increasing and contain zero.
  """
  require_causal_window: bool = False
  """Reject any positive actor offset. Enabled by online-teleoperation tasks."""
  critic_window_offsets: tuple[int, ...] | None = None
  """Training-only privileged future offsets; all must be positive and increasing."""
  reconstruction_window_offsets: tuple[int, ...] | None = None
  """Sparse future targets for the training-only intent reconstruction head.

  The causal task uses ``(+5, +10, +20)`` following TeleGate
  (arXiv:2602.09628). These targets never enter actor observations or deployment.
  """
  heading_closed_loop: bool = False
  """Append SONIC-style closed-loop relative pelvis orientation to every token.

  False preserves the Extreme-RGMT command exactly at 38 channels per token. True
  appends the first two rotation-matrix columns of
  ``q_robot_current^-1 * q_ref_future``, producing 44 channels per token.
  """

  recovery_probability: float = 0.0
  """Probability that a resetting environment becomes a recovery environment.
  RGMT (arXiv:2601.23080v1) Sec. II-D uses 0.15; Extreme-RGMT inherits the mechanism
  without restating it. 0 disables fall-recovery training entirely -- the play,
  evaluation and stratification paths rely on that."""
  recovery_window_s: float = 3.0
  """Recovery window during which instability terminations are suspended
  (RGMT Sec. II-D: "a predetermined recovery window of 3 seconds")."""
  recovery_assist_force_range: tuple[float, float] = (0.0, 200.0)
  """Upward assistance-force magnitude range in newtons (RGMT Sec. II-D)."""
  recovery_assist_anneal_steps: int = 2_400_000
  """Env steps over which the assistance force anneals linearly to zero. RGMT says
  "linearly annealed over training iterations" without a number; 2.4M env steps is
  100k iterations x 24 rollout steps, the nominal Stage-I run. ASSUMPTION -- it has to
  track ``rl_cfgs.stage1_runner_cfg``'s ``max_iterations``: anneal shorter than the run
  means the assistance is gone for most of training, longer means it never reaches 0.

  The clock is ``MultiMotionCommand.recovery_steps_elapsed`` -- steps taken with
  ``recovery_probability > 0`` -- which the runner checkpoints, so resuming a
  recovery run continues the same schedule while enabling recovery on top of a
  checkpoint that never had it starts the schedule from zero. RGMT trains recovery
  in a single end-to-end run, where the two coincide."""
  recovery_root_height_range: tuple[float, float] = (0.35, 0.65)
  """Pelvis-height range in metres for randomized recovery poses. ASSUMPTION: RGMT
  publishes no pose distribution; this low range yields contact-rich starts without
  requiring any recovery motion data."""
  recovery_root_tilt_range: tuple[float, float] = (
    math.pi / 3.0,
    2.0 * math.pi / 3.0,
  )
  """Absolute root tilt about a random horizontal axis, radians. ASSUMPTION: the
  60--120 degree range excludes upright starts while avoiding exclusively inverted
  configurations; yaw is sampled uniformly over the full circle."""
  recovery_joint_position_jitter: tuple[float, float] = (-0.25, 0.25)
  """Per-joint offset around the ordinary reference pose, radians. ASSUMPTION: values
  are clipped to the robot's soft limits and initial velocities are zero."""

  debug_vis: bool = True
  """Show the reference motion as a green semi-transparent ghost robot."""

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)
