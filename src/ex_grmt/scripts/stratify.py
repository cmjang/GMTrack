"""Motion stratification (paper Sec. IV-C).

After Stage I, every clip is scored with five randomized rollouts of the base policy.
Clips completing at least 80 % of the time form the **mastered** set ``D_m``; the rest
form the **challenging** set ``D_c``. Stage II consumes the two as disjoint
consolidation / acquisition pools.

Also emits the Fig. 4 kinematic statistics (99th-percentile per motion, then median
over each set) so the two sets can be compared the way the paper does.

Usage::

    uv run python -m ex_grmt.scripts.stratify \\
        --checkpoint logs/rsl_rl/ex_grmt_stage1/<run>/model_29999.pt
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mjlab
import numpy as np
import torch
import tyro

from ex_grmt.scripts._harness import build_env_and_policy, resolve_device
from ex_grmt.scripts.rollout_eval import rollout_clips


@dataclass
class Config:
  checkpoint: str
  """Stage-I checkpoint."""
  task: str = "ExGRMT-Stage1-Flat-Unitree-G1"
  manifest: str = "data/manifests/all.json"
  out_dir: str = "data/manifests"
  num_envs: int = 256
  trials: int = 5
  """Randomized rollouts per clip (paper Sec. IV-C)."""
  threshold: float = 0.8
  """Completion rate at or above which a clip counts as mastered."""
  device: str | None = None


def _kinematic_stats(lib, clip_id: int) -> dict[str, float]:
  """Fig. 4 axes, as the 99th percentile over the clip's frames."""
  start = int(lib.clip_start[clip_id])
  stop = start + int(lib.clip_len[clip_id])
  root_lin = lib.body_lin_vel_w[start:stop, 0]
  root_ang = lib.body_ang_vel_w[start:stop, 0]
  joint_vel = lib.joint_vel[start:stop]
  dt = 1.0 / lib.fps

  root_speed = root_lin.norm(dim=-1)
  root_accel = (root_lin[1:] - root_lin[:-1]).norm(dim=-1) / dt
  # Airborne = every tracked body above a small ground clearance.
  min_body_z = lib.body_pos_w[start:stop, :, 2].min(dim=-1).values

  def p99(x: torch.Tensor) -> float:
    return float(torch.quantile(x.flatten().float(), 0.99))

  return {
    "root_linear_speed": p99(root_speed),
    "root_angular_speed": p99(root_ang.norm(dim=-1)),
    "root_linear_accel": p99(root_accel) if root_accel.numel() else 0.0,
    "com_vertical_speed": p99(root_lin[:, 2].abs()),
    "joint_velocity": p99(joint_vel.abs()),
    "airborne_ratio": float((min_body_z > 0.08).float().mean()),
  }


def main(cfg: Config) -> None:
  device = resolve_device(cfg.device)
  env, policy, command = build_env_and_policy(
    task_id=cfg.task,
    checkpoint=cfg.checkpoint,
    num_envs=cfg.num_envs,
    device=device,
    # Domain randomization stays on: that is what makes the rollouts "randomized".
    play=False,
    manifest=cfg.manifest,
  )
  lib = command.lib
  clip_ids = torch.arange(lib.num_clips, dtype=torch.long, device=device)

  print(f"[ex-grmt] stratifying {lib.num_clips} clips x {cfg.trials} trials")
  results = rollout_clips(env, policy, command, clip_ids, trials=cfg.trials)

  with Path(cfg.manifest).open() as f:
    entries = {c["name"]: c for c in json.load(f)["clips"]}

  mastered, challenging, report = [], [], {}
  for clip_id, res in sorted(results.items()):
    entry = entries[res.name]
    target = mastered if res.completion_rate >= cfg.threshold else challenging
    target.append(entry)
    report[res.name] = {
      **res.summary(),
      **_kinematic_stats(lib, clip_id),
      "source": entry.get("source", "unknown"),
    }

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for name, clips in (("mastered", mastered), ("challenging", challenging)):
    with (out_dir / f"{name}.json").open("w") as f:
      json.dump({"clips": clips}, f, indent=2)
  with (out_dir / "stratification_report.json").open("w") as f:
    json.dump(report, f, indent=2)

  def hours(clips: list[dict]) -> float:
    return sum(c["num_frames"] / c["fps"] for c in clips) / 3600.0

  print(
    f"[ex-grmt] mastered   D_m: {len(mastered):5d} clips, {hours(mastered):.2f} h\n"
    f"[ex-grmt] challenging D_c: {len(challenging):5d} clips, {hours(challenging):.2f} h"
  )

  # Fig. 4: median of the per-motion 99th percentiles, per set.
  for label, clips in (("mastered", mastered), ("challenging", challenging)):
    if not clips:
      continue
    keys = list(_kinematic_stats(lib, 0).keys())
    medians = {
      k: float(np.median([report[c["name"]][k] for c in clips])) for k in keys
    }
    print(f"[ex-grmt] {label:12s} " + "  ".join(f"{k}={v:.2f}" for k, v in medians.items()))

  env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
