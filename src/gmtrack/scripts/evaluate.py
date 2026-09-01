"""Tracking evaluation (paper Sec. VI-A).

Reports the four metrics used in Tables VI/VII:

* ``Succ.``      -- fraction of rollouts completing the reference (root height within
                    0.2 m of the reference throughout)
* ``E_MPJPE``    -- root-relative mean per-joint position error, mm
* ``d_vel``      -- joint velocity error, mm/frame
* ``d_acc``      -- joint acceleration error, mm/frame^2

Usage::

    uv run python -m gmtrack.scripts.evaluate \\
        --checkpoint logs/rsl_rl/gmtrack_stage2/<run>/model_49999.pt \\
        --manifest data/current/<stratification-run>/challenging.json
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import numpy as np
import torch
import tyro

from gmtrack.protocol import validate_evaluation_manifest
from gmtrack.provenance import PAPER_ID, PAPER_SHA256, sha256_file
from gmtrack.scripts._harness import build_env_and_policy, resolve_device
from gmtrack.scripts.rollout_eval import rollout_clips


@dataclass
class Config:
  checkpoint: str
  training_seed: int | None = None
  """Independent training seed represented by this checkpoint."""
  manifest: str = (
    "data/current/stage1-116500-final30-cartwheel-balanced-nofall-probe/stratified.json"
  )
  task: str = "GMTrack-Stage2-Flat-Unitree-G1"
  num_envs: int = 256
  rollouts_per_clip: int = 1
  """Rollouts per clip. The paper's "five random seeds" are five *training* seeds,
  a between-checkpoint variance this script cannot produce; with the clean play
  environment and a deterministic policy, repeated trials of one checkpoint are
  identical, so the default is 1. Raise it only if you add stochasticity."""
  trials: int | None = None
  """Deprecated CLI alias for ``rollouts_per_clip``."""
  eval_mode: Literal["nominal", "randomized"] = "nominal"
  """Nominal for paper metrics; randomized for an explicit robustness probe."""
  device: str | None = None
  out: str | None = None
  """Optional JSON path for the per-clip breakdown."""
  group_by_source: bool = True
  """Also report per-source aggregates (in-source vs unseen)."""


def main(cfg: Config) -> None:
  stratification_validation = validate_evaluation_manifest(cfg.manifest)
  rollouts_per_clip = cfg.rollouts_per_clip
  if cfg.trials is not None:
    if cfg.rollouts_per_clip != 1:
      raise ValueError("Pass only --rollouts-per-clip; --trials is deprecated.")
    warnings.warn(
      "--trials is deprecated; use --rollouts-per-clip",
      DeprecationWarning,
      stacklevel=2,
    )
    rollouts_per_clip = cfg.trials

  device = resolve_device(cfg.device)
  env, policy, command = build_env_and_policy(
    task_id=cfg.task,
    checkpoint=cfg.checkpoint,
    num_envs=cfg.num_envs,
    device=device,
    manifest=cfg.manifest,
    eval_mode=cfg.eval_mode,
  )
  lib = command.lib
  clip_ids = torch.arange(lib.num_clips, dtype=torch.long, device=device)

  print(
    f"[gmtrack] evaluating {lib.num_clips} clips x "
    f"{rollouts_per_clip} rollouts ({cfg.eval_mode})"
  )
  results = rollout_clips(
    env, policy, command, clip_ids, rollouts_per_clip=rollouts_per_clip
  )

  per_clip = {res.name: res.summary() for res in results.values()}

  def aggregate(names: list[str]) -> dict[str, float | int]:
    if not names:
      return {}
    metric_names = [n for n in names if int(per_clip[n]["finite_metric_steps"]) > 0]
    if not metric_names:
      raise ValueError("No finite metric prefix exists for any requested clip.")
    return {
      "succ_pct": 100.0
      * float(np.mean([per_clip[n]["completion_rate"] for n in names])),
      "mpjpe_mm": float(np.mean([per_clip[n]["mpjpe_mm"] for n in metric_names])),
      "d_vel": float(np.mean([per_clip[n]["d_vel"] for n in metric_names])),
      "d_acc": float(np.mean([per_clip[n]["d_acc"] for n in metric_names])),
      "nonfinite_failures": int(
        sum(int(per_clip[n]["nonfinite_failures"]) for n in names)
      ),
      "metric_clips": len(metric_names),
      "clips": len(names),
    }

  overall = aggregate(list(per_clip))
  print(
    f"\n{'set':<16}{'clips':>7}{'Succ.(%)':>11}{'E_MPJPE(mm)':>14}"
    f"{'d_vel':>10}{'d_acc':>10}{'NonFinite':>11}"
  )
  print("-" * 79)

  def show(label: str, agg: dict[str, float | int]) -> None:
    if not agg:
      return
    print(
      f"{label:<16}{agg['clips']:>7}{agg['succ_pct']:>11.2f}"
      f"{agg['mpjpe_mm']:>14.2f}{agg['d_vel']:>10.2f}{agg['d_acc']:>10.2f}"
      f"{int(agg['nonfinite_failures']):>11}"
    )

  show("ALL", overall)

  by_source: dict[str, dict[str, float | int]] = {}
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
          "checkpoint_sha256": sha256_file(cfg.checkpoint),
          "manifest": cfg.manifest,
          "manifest_sha256": sha256_file(cfg.manifest),
          "paper": {"id": PAPER_ID, "sha256": PAPER_SHA256},
          "stratification_validation": stratification_validation,
          "training_seed": cfg.training_seed,
          "eval_mode": cfg.eval_mode,
          "rollouts_per_clip": rollouts_per_clip,
          "overall": overall,
          "by_source": by_source,
          "per_clip": per_clip,
        },
        f,
        indent=2,
      )
    print(f"\n[gmtrack] wrote {out}")

  env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
