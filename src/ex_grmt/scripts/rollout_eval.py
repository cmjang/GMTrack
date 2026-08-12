"""Shared per-clip rollout harness for stratification and evaluation.

Both `stratify.py` (Sec. IV-C) and `evaluate.py` (Sec. VI-A) need the same thing:
drive a trained policy over every clip in a manifest and score each clip
independently. This module owns that loop so the two scripts cannot drift apart on
what "success" means.

Success criterion (paper Sec. VI-A): a rollout is unsuccessful when the robot root
height deviates from the reference by more than 0.2 m -- and *only* then. The
training failure terminations (end-effector z-deviation, anchor position and
orientation) are removed by `_harness.build_env_and_policy`, because they trip far
earlier than the paper's criterion and would deflate Succ. below anything
comparable to Table VI.

Non-finite reference, robot or derived metric state is handled separately as a data
integrity failure. Such a rollout cannot count toward Succ., but this does not add a
second motion-quality criterion or re-enable any training termination; finite metric
steps before the corruption are retained for diagnosis.

Metric spaces (paper Sec. VI-A): E_MPJPE, d_vel and d_acc are all root-relative
*Cartesian* per-body quantities in mm, mm/frame and mm/frame^2. d_vel/d_acc are the
first and second temporal differences of the same per-body position error that
E_MPJPE averages -- a first difference of the error equals the velocity error, since
(ref_t - rob_t) - (ref_{t-1} - rob_{t-1}) = v_ref - v_rob. They are *not* computed
on joint angles; that would report milliradians and overstate the paper's
mm-denominated numbers several-fold.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch

from ex_grmt.mdp.commands import MultiMotionCommand

ROOT_HEIGHT_TOLERANCE = 0.2
"""Metres. Paper Sec. VI-A."""

_REFERENCE_STATE_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_ROBOT_STATE_FIELDS = (
  "robot_joint_pos",
  "robot_joint_vel",
  "robot_body_pos_w",
  "robot_body_quat_w",
  "robot_body_lin_vel_w",
  "robot_body_ang_vel_w",
)
_PHYSICS_STATE_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "sensordata",
)


def _finite_rows(
  tensors: tuple[torch.Tensor, ...], n: int, device: torch.device
) -> torch.Tensor:
  """Return a per-environment mask that is finite across every supplied tensor."""
  finite = torch.ones(n, dtype=torch.bool, device=device)
  for value in tensors:
    finite &= torch.isfinite(value[:n]).reshape(n, -1).all(dim=1)
  return finite


def _command_state_is_finite(
  command: MultiMotionCommand, fields: tuple[str, ...], n: int
) -> torch.Tensor:
  """Check all available reference or robot state channels for each environment."""
  values = tuple(
    value
    for name in fields
    if isinstance((value := getattr(command, name, None)), torch.Tensor)
  )
  return _finite_rows(values, n, command.device)


def _physics_state_is_finite(env, n: int, device: torch.device) -> torch.Tensor:
  """Mirror the training guard's MuJoCo-state coverage without terminating early."""
  sim_data = getattr(env.unwrapped.sim, "data", None)
  finite = torch.ones(n, dtype=torch.bool, device=device)
  for name in _PHYSICS_STATE_FIELDS:
    value = getattr(sim_data, name, None)
    if isinstance(value, torch.Tensor):
      # Match nonfinite_physics_state exactly; this also supports simulator fields
      # represented as one flat vector rather than an explicit leading env axis.
      finite &= torch.isfinite(value).reshape(env.num_envs, -1)[:n].all(dim=1)
  return finite


def _reference_state_is_finite(command: MultiMotionCommand, n: int) -> torch.Tensor:
  return _command_state_is_finite(command, _REFERENCE_STATE_FIELDS, n)


def _robot_state_is_finite(env, command: MultiMotionCommand, n: int) -> torch.Tensor:
  command_finite = _command_state_is_finite(command, _ROBOT_STATE_FIELDS, n)
  return command_finite & _physics_state_is_finite(env, n, command.device)


def _transition_horizons(clip_lengths: torch.Tensor) -> torch.Tensor:
  """Number of state transitions in each sampled reference sequence."""
  if bool((clip_lengths < 2).any()):
    raise ValueError("Every evaluated clip must contain at least two frames.")
  return clip_lengths - 1


def _cartesian_error_metrics(
  err: torch.Tensor, prev_err: torch.Tensor, prev_vel_err: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Per-step E_MPJPE / d_vel / d_acc contributions from the body-position error.

  Args:
    err: ``(N, B, 3)`` root-relative reference-minus-robot body positions, metres.
    prev_err: Same quantity one control step earlier. At reset the robot is snapped
      onto the reference in both pose and velocity, so zeros are the exact
      initial state, not a fabricated one.
    prev_vel_err: Previous step's first difference (velocity error), metres/frame.

  Returns:
    ``(mpjpe_mm, d_vel, d_acc, vel_err)`` -- the first three are ``(N,)`` per-body
    means in mm, mm/frame and mm/frame^2; ``vel_err`` is the ``(N, B, 3)`` first
    difference to feed back as the next step's ``prev_vel_err``.
  """
  vel_err = err - prev_err
  acc_err = vel_err - prev_vel_err
  mpjpe_mm = torch.norm(err, dim=-1).mean(dim=-1) * 1000.0
  d_vel = torch.norm(vel_err, dim=-1).mean(dim=-1) * 1000.0
  d_acc = torch.norm(acc_err, dim=-1).mean(dim=-1) * 1000.0
  return mpjpe_mm, d_vel, d_acc, vel_err


def _initialize_clip_observations(
  env, command: MultiMotionCommand, env_ids: torch.Tensor
):
  """Build frame-0 histories, then return the frame-1 policy observation.

  ``env.reset()`` computed and cached observations for whatever clip the command had
  before ``set_clip``. Resetting the manager backfills every history slot from the
  selected clip's frame 0. Frame 0 is the robot initial condition, so the command is
  then advanced to frame 1 without advancing proprio/action history a second time.
  """
  raw_env = env.unwrapped
  raw_env.sim.forward()
  command.update_relative_body_poses()
  raw_env.observation_manager.reset(env_ids)
  raw_env.sim.sense()
  raw_env.obs_buf = raw_env.observation_manager.compute(update_history=True)

  command.time_steps[env_ids] += 1
  command.update_relative_body_poses()
  # ObservationManager has no public cache-invalidation method that preserves the
  # histories we just initialized. Its reset() cannot be used here because it would
  # erase them, so invalidate only the documented compute cache.
  raw_env.observation_manager._obs_buffer = None
  raw_env.obs_buf = raw_env.observation_manager.compute(update_history=False)
  return env.get_observations()


@dataclass
class ClipResult:
  """Per-clip aggregate over all trials."""

  name: str
  trials: int = 0
  successes: int = 0
  nonfinite_failures: int = 0
  mpjpe_mm: list[float] = field(default_factory=list)
  vel_err: list[float] = field(default_factory=list)
  acc_err: list[float] = field(default_factory=list)

  @property
  def completion_rate(self) -> float:
    return self.successes / self.trials if self.trials else 0.0

  def summary(self) -> dict[str, float | int]:
    def mean(xs: list[float]) -> float:
      finite = [x for x in xs if math.isfinite(x)]
      if not finite:
        return 0.0
      scale = max(abs(x) for x in finite)
      if scale == 0.0:
        return 0.0
      # Scaling keeps the aggregate finite even if a caller constructs a result
      # with many extremely large (but individually finite) metric values.
      return float(scale * math.fsum(x / scale for x in finite) / len(finite))

    return {
      "completion_rate": self.completion_rate,
      "nonfinite_failures": self.nonfinite_failures,
      "finite_metric_steps": len(self.mpjpe_mm),
      "mpjpe_mm": mean(self.mpjpe_mm),
      "d_vel": mean(self.vel_err),
      "d_acc": mean(self.acc_err),
    }


@torch.no_grad()
def rollout_clips(
  env,
  policy,
  command: MultiMotionCommand,
  clip_ids: torch.Tensor,
  rollouts_per_clip: int = 1,
  *,
  trials: int | None = None,
) -> dict[int, ClipResult]:
  """Run each clip ``rollouts_per_clip`` times, batched across environments.

  Clips are processed in chunks of ``env.num_envs``. Each chunk runs for the longest
  clip in it; environments whose clip is shorter, or which have already failed, are
  frozen (their metrics stop accumulating) rather than being allowed to auto-reset
  into an unrelated clip.

  Returns:
    ``{library clip id: ClipResult}``.
  """
  if trials is not None:
    if rollouts_per_clip != 1:
      raise ValueError("Pass only rollouts_per_clip; trials is a deprecated alias.")
    warnings.warn(
      "trials is deprecated; use rollouts_per_clip",
      DeprecationWarning,
      stacklevel=2,
    )
    rollouts_per_clip = trials
  if rollouts_per_clip < 1:
    raise ValueError("rollouts_per_clip must be at least one.")

  lib = command.lib
  device = command.device
  num_envs = env.num_envs

  results: dict[int, ClipResult] = {
    int(c): ClipResult(name=lib.clips[int(c)].name) for c in clip_ids
  }

  jobs = [(int(c), rollout) for c in clip_ids for rollout in range(rollouts_per_clip)]

  for start in range(0, len(jobs), num_envs):
    chunk = jobs[start : start + num_envs]
    n = len(chunk)
    env_ids = torch.arange(n, device=device)
    chunk_clips = torch.tensor([c for c, _ in chunk], dtype=torch.long, device=device)

    env.reset()
    command.set_clip(env_ids, chunk_clips, frames=0)

    # Capture the true frame-0 error before advancing the command. It is normally
    # zero, but set_clip may clip a reference joint at the robot's soft limit.
    ref_rel_0 = command.body_pos_w[:n] - command.body_pos_w[:n, :1]
    rob_rel_0 = command.robot_body_pos_w[:n] - command.robot_body_pos_w[:n, :1]
    prev_err = ref_rel_0 - rob_rel_0
    prev_vel_err = torch.zeros_like(prev_err)
    initial_finite = (
      _reference_state_is_finite(command, n)
      & _robot_state_is_finite(env, command, n)
      & _finite_rows((prev_err,), n, device)
    )

    obs = _initialize_clip_observations(env, command, env_ids)

    # Frame 0 is installed before stepping, so an N-frame reference contains N-1
    # transitions. Running N steps makes MultiMotionCommand wrap to a new clip and
    # contaminates the final metric with an unrelated reference frame.
    horizons = _transition_horizons(lib.clip_len[chunk_clips])
    horizon = int(horizons.max().item())

    alive = initial_finite.clone()
    nonfinite = ~initial_finite
    # Accumulate in float64 so a long finite prefix cannot overflow the JSON-facing
    # aggregate even when an individual float32 norm is close to its upper bound.
    pos_err_sum = torch.zeros(n, dtype=torch.float64, device=device)
    pos_err_count = torch.zeros(n, dtype=torch.float64, device=device)
    vel_err_sum = torch.zeros(n, dtype=torch.float64, device=device)
    acc_err_sum = torch.zeros(n, dtype=torch.float64, device=device)

    for step in range(horizon):
      within_clip = step < horizons

      # command_manager.compute advances the command at the end of env.step(). Save
      # the reference that produced this action so the resulting robot state is not
      # compared to the following frame.
      target_pos = command.body_pos_w[:n].clone()
      target_rel = target_pos - target_pos[:, :1]
      target_root_z = target_pos[:, 0, 2].clone()

      eligible = alive & within_clip
      pre_step_finite = (
        _reference_state_is_finite(command, n)
        & _robot_state_is_finite(env, command, n)
        & _finite_rows((target_pos, target_rel, target_root_z), n, device)
      )
      pre_step_nonfinite = eligible & ~pre_step_finite
      nonfinite |= pre_step_nonfinite
      alive &= ~pre_step_nonfinite

      # rsl-rl's inference policy is the actor module itself; it does not install a
      # no-grad context. Passing graph-attached actions into mjlab makes the next
      # observation-history update fail because its in-place circular buffer cannot
      # accept tensors that require gradients.
      with torch.inference_mode():
        actions = policy(obs)
      action_finite = _finite_rows((actions,), n, device)
      action_nonfinite = alive & within_clip & ~action_finite
      nonfinite |= action_nonfinite
      alive &= ~action_nonfinite
      # Failed/completed environments still participate in vectorized env.step().
      # Zeroing their actions prevents a corrupt observation or action from feeding
      # another NaN into MuJoCo while the rest of the chunk finishes.
      step_active = alive & within_clip
      actions = actions.clone()
      actions[:n] = torch.where(
        step_active.view(n, *([1] * (actions.ndim - 1))),
        actions[:n],
        torch.zeros_like(actions[:n]),
      )
      obs, _, dones, _ = env.step(actions)

      if bool((dones[:n].bool() & within_clip).any()):
        raise RuntimeError(
          "Environment terminated during a harness-owned clip rollout. Evaluation "
          "must remove sequence/failure terminations; success is judged solely by "
          "the root-height criterion (paper Sec. VI-A)."
        )

      # Root-relative Cartesian body-position error and its first/second temporal
      # differences (paper Sec. VI-A metric definitions). "Root-relative" = each
      # side's body positions minus its *own* root translation (body index 0 is the
      # pelvis). The training-side `body_pos_relative_w` is NOT that quantity: it
      # yaw-aligns and horizontally re-anchors the reference to the robot while
      # keeping reference height, which retains a root-height error term and makes
      # the numbers incomparable with Table VI.
      rob_rel = command.robot_body_pos_w[:n] - command.robot_body_pos_w[:n, :1]
      err = target_rel - rob_rel
      per_joint, d_vel, d_acc, vel_err = _cartesian_error_metrics(
        err, prev_err, prev_vel_err
      )
      step_finite = _robot_state_is_finite(env, command, n) & _finite_rows(
        (err, per_joint, d_vel, d_acc, vel_err), n, device
      )
      step_nonfinite = alive & within_clip & ~step_finite
      nonfinite |= step_nonfinite
      alive &= ~step_nonfinite

      counting = alive & within_clip
      pos_err_sum += torch.where(
        counting, per_joint.double(), torch.zeros_like(pos_err_sum)
      )
      vel_err_sum += torch.where(
        counting, d_vel.double(), torch.zeros_like(vel_err_sum)
      )
      acc_err_sum += torch.where(
        counting, d_acc.double(), torch.zeros_like(acc_err_sum)
      )
      pos_err_count += counting.double()

      state_mask = counting.view(n, *([1] * (err.ndim - 1)))
      prev_err = torch.where(state_mask, err, prev_err)
      prev_vel_err = torch.where(state_mask, vel_err, prev_vel_err)

      height_dev = (target_root_z - command.robot_body_pos_w[:n, 0, 2]).abs()
      failed = counting & (height_dev > ROOT_HEIGHT_TOLERANCE)
      alive &= ~(failed & within_clip)

      if not bool(alive.any()):
        break

    for i, (clip_id, _) in enumerate(chunk):
      res = results[clip_id]
      res.trials += 1
      if bool(alive[i]):
        res.successes += 1
      if bool(nonfinite[i]):
        res.nonfinite_failures += 1
      if bool(pos_err_count[i] > 0):
        count = pos_err_count[i]
        res.mpjpe_mm.append(float(pos_err_sum[i] / count))
        res.vel_err.append(float(vel_err_sum[i] / count))
        res.acc_err.append(float(acc_err_sum[i] / count))

  return results
