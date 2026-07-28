"""Shared per-clip rollout harness for stratification and evaluation.

Both `stratify.py` (Sec. IV-C) and `evaluate.py` (Sec. VI-A) need the same thing:
drive a trained policy over every clip in a manifest and score each clip
independently. This module owns that loop so the two scripts cannot drift apart on
what "success" means.

Success criterion (paper Sec. VI-A): a rollout is unsuccessful when the robot root
height deviates from the reference by more than 0.2 m. Note this is *stricter* than
the training termination (`bad_anchor_pos_z_only`, threshold 0.25 on the anchor body),
so it is evaluated explicitly here rather than inferred from terminations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ex_grmt.mdp.commands import MultiMotionCommand

ROOT_HEIGHT_TOLERANCE = 0.2
"""Metres. Paper Sec. VI-A."""


@dataclass
class ClipResult:
  """Per-clip aggregate over all trials."""

  name: str
  trials: int = 0
  successes: int = 0
  mpjpe_mm: list[float] = field(default_factory=list)
  vel_err: list[float] = field(default_factory=list)
  acc_err: list[float] = field(default_factory=list)

  @property
  def completion_rate(self) -> float:
    return self.successes / self.trials if self.trials else 0.0

  def summary(self) -> dict[str, float]:
    def mean(xs: list[float]) -> float:
      return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
      "completion_rate": self.completion_rate,
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
  trials: int = 1,
) -> dict[int, ClipResult]:
  """Run every clip in ``clip_ids`` ``trials`` times, batched across environments.

  Clips are processed in chunks of ``env.num_envs``. Each chunk runs for the longest
  clip in it; environments whose clip is shorter, or which have already failed, are
  frozen (their metrics stop accumulating) rather than being allowed to auto-reset
  into an unrelated clip.

  Returns:
    ``{library clip id: ClipResult}``.
  """
  lib = command.lib
  device = command.device
  num_envs = env.num_envs

  results: dict[int, ClipResult] = {
    int(c): ClipResult(name=lib.clips[int(c)].name) for c in clip_ids
  }

  jobs = [(int(c), t) for c in clip_ids for t in range(trials)]

  for start in range(0, len(jobs), num_envs):
    chunk = jobs[start : start + num_envs]
    n = len(chunk)
    env_ids = torch.arange(n, device=device)
    chunk_clips = torch.tensor([c for c, _ in chunk], dtype=torch.long, device=device)

    obs, _ = env.reset()
    command.set_clip(env_ids, chunk_clips, frames=0)
    env.unwrapped.sim.forward()
    command.update_relative_body_poses()
    obs = env.get_observations()

    lengths = lib.clip_len[chunk_clips]
    horizon = int(lengths.max().item())

    alive = torch.ones(n, dtype=torch.bool, device=device)
    pos_err_sum = torch.zeros(n, device=device)
    pos_err_count = torch.zeros(n, device=device)
    vel_err_sum = torch.zeros(n, device=device)
    acc_err_sum = torch.zeros(n, device=device)

    prev_joint_err = torch.zeros(n, command.robot_joint_pos.shape[1], device=device)
    prev_joint_vel_err = torch.zeros_like(prev_joint_err)

    for step in range(horizon):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)

      in_range = torch.arange(n, device=device)
      within_clip = step < lengths

      # Root-relative per-joint position error, in millimetres.
      ref_body = command.body_pos_relative_w[:n]
      cur_body = command.robot_body_pos_w[:n]
      per_joint = torch.norm(ref_body - cur_body, dim=-1).mean(dim=-1) * 1000.0

      # Joint-space first/second differences of the tracking error (d_vel, d_acc).
      joint_err = (command.joint_pos[:n] - command.robot_joint_pos[:n]) * 1000.0
      joint_vel_err = joint_err - prev_joint_err
      joint_acc_err = joint_vel_err - prev_joint_vel_err
      prev_joint_err = joint_err
      prev_joint_vel_err = joint_vel_err

      counting = alive & within_clip
      pos_err_sum += torch.where(counting, per_joint, torch.zeros_like(per_joint))
      vel_err_sum += torch.where(
        counting, joint_vel_err.abs().mean(-1), torch.zeros_like(per_joint)
      )
      acc_err_sum += torch.where(
        counting, joint_acc_err.abs().mean(-1), torch.zeros_like(per_joint)
      )
      pos_err_count += counting.float()

      height_dev = (
        command.body_pos_w[:n, 0, 2] - command.robot_body_pos_w[:n, 0, 2]
      ).abs()
      failed = (height_dev > ROOT_HEIGHT_TOLERANCE) | dones[:n].bool()
      alive &= ~(failed & within_clip)

      del in_range
      if not bool(alive.any()):
        break

    counts = pos_err_count.clamp(min=1.0)
    for i, (clip_id, _) in enumerate(chunk):
      res = results[clip_id]
      res.trials += 1
      if bool(alive[i]):
        res.successes += 1
      res.mpjpe_mm.append(float(pos_err_sum[i] / counts[i]))
      res.vel_err.append(float(vel_err_sum[i] / counts[i]))
      res.acc_err.append(float(acc_err_sum[i] / counts[i]))

  return results
