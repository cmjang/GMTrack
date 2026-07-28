"""Tracking evaluation (paper Sec. VI-A).

Reports the four metrics used in Tables VI/VII:

* ``Succ.``      -- fraction of rollouts completing the reference (root height within
                    0.2 m of the reference throughout)
* ``E_MPJPE``    -- root-relative mean per-joint position error, mm
* ``d_vel``      -- joint velocity error, mm/frame
* ``d_acc``      -- joint acceleration error, mm/frame^2

Usage::

    uv run python -m ex_grmt.scripts.evaluate \\
        --checkpoint logs/rsl_rl/ex_grmt_stage2/<run>/model_49999.pt \\
        --manifest data/manifests/challenging.json
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
  manifest: str = "data/manifests/all.json"
  task: str = "ExGRMT-Stage2-Flat-Unitree-G1"
  num_envs: int = 256
  trials: int = 5
  """Paper averages over five random seeds; here, five rollouts per clip."""
  device: str | None = None
  out: str | None = None
  """Optional JSON path for the per-clip breakdown."""
  group_by_source: bool = True
  """Also report per-source aggregates (in-source vs unseen)."""


def main(cfg: Config) -> None:
  device = resolve_device(cfg.device)
  env, policy, command = build_env_and_policy(
    task_id=cfg.task,
    checkpoint=cfg.checkpoint,
    num_envs=cfg.num_envs,
    device=device,
    play=True,
    manifest=cfg.manifest,
  )
  lib = command.lib
  clip_ids = torch.arange(lib.num_clips, dtype=torch.long, device=device)

  print(f"[ex-grmt] evaluating {lib.num_clips} clips x {cfg.trials} trials")
  results = rollout_clips(env, policy, command, clip_ids, trials=cfg.trials)

  per_clip = {res.name: res.summary() for res in results.values()}

  def aggregate(names: list[str]) -> dict[str, float]:
    if not names:
      return {}
    return {
      "succ_pct": 100.0 * float(np.mean([per_clip[n]["completion_rate"] for n in names])),
      "mpjpe_mm": float(np.nanmean([per_clip[n]["mpjpe_mm"] for n in names])),
      "d_vel": float(np.nanmean([per_clip[n]["d_vel"] for n in names])),
      "d_acc": float(np.nanmean([per_clip[n]["d_acc"] for n in names])),
      "clips": len(names),
    }

  overall = aggregate(list(per_clip))
  print(
    f"\n{'set':<16}{'clips':>7}{'Succ.(%)':>11}{'E_MPJPE(mm)':>14}"
    f"{'d_vel':>10}{'d_acc':>10}"
  )
  print("-" * 68)

  def show(label: str, agg: dict[str, float]) -> None:
    if not agg:
      return
    print(
      f"{label:<16}{agg['clips']:>7}{agg['succ_pct']:>11.2f}"
      f"{agg['mpjpe_mm']:>14.2f}{agg['d_vel']:>10.2f}{agg['d_acc']:>10.2f}"
    )

  show("ALL", overall)

  by_source: dict[str, dict[str, float]] = {}
  if cfg.group_by_source:
    sources: dict[str, list[str]] = {}
    for info in lib.clips:
      sources.setdefault(info.source, []).append(info.name)
    for source, names in sorted(sources.items()):
      by_source[source] = aggregate(names)
      show(source, by_source[source])

  if cfg.out:
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
      json.dump(
        {
          "checkpoint": cfg.checkpoint,
          "manifest": cfg.manifest,
          "overall": overall,
          "by_source": by_source,
          "per_clip": per_clip,
        },
        f,
        indent=2,
      )
    print(f"\n[ex-grmt] wrote {out}")

  env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
