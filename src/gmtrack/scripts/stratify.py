"""Post-Stage-I motion slicing and stratification (paper Sec. IV-C).

After Stage I, sequences longer than 10 seconds are represented as logical 10-second
clips (shorter sequences and every trailing remainder are retained). Every resulting
clip is then scored with five randomized rollouts of the base policy. Clips completing
at least 80 % of the time form the **mastered** set ``D_m``; the rest form the
**challenging** set ``D_c``. Stage II consumes the two as disjoint consolidation /
acquisition pools.

Also emits the Fig. 4 kinematic statistics (99th-percentile per motion, then median
over each set) so the two sets can be compared the way the paper does.

Usage::

    uv run python -m gmtrack.scripts.stratify \\
        --checkpoint logs/rsl_rl/gmtrack_stage1/<run>/model_29999.pt
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab
import numpy as np
import torch
import tyro
from mjlab.utils import random as random_utils

from gmtrack.protocol import (
  STRATIFICATION_MAX_CLIP_SECONDS,
  STRATIFICATION_SCHEMA,
  STRATIFICATION_SCHEMA_VERSION,
  STRATIFICATION_THRESHOLD,
  STRATIFICATION_TRIALS,
  artifact_set_sha256,
  make_stratification_protocol,
  make_stratification_provenance,
  validate_stage2_manifests,
)
from gmtrack.scripts._harness import build_env_and_policy, resolve_device
from gmtrack.scripts.rollout_eval import rollout_clips


@dataclass
class Config:
  checkpoint: str
  """Stage-I checkpoint."""
  task: str = "GMTrack-Stage1-Flat-Unitree-G1"
  manifest: str = (
    "data/current/"
    "stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"
  )
  """Stage-I manifest containing complete source sequences."""
  stratified_manifest: str = "data/current/stratification/stratified.json"
  """Logical <=10-second clip manifest produced before evaluation."""
  out_dir: str = "data/current/stratification"
  clip_seconds: float = 10.0
  """Paper Sec. IV-C stratification clip duration."""
  num_envs: int = 256
  trials: int = 5
  """Randomized rollouts per clip (paper Sec. IV-C)."""
  threshold: float = 0.8
  """Completion rate at or above which a clip counts as mastered."""
  seed: int = 0
  """Seed for domain randomization, external pushes, and observation noise."""
  dataset_label: str = "v1-proxy"
  """Honest label for the public proxy replacing unavailable paper data."""
  device: str | None = None


def _frame_ranges(num_frames: int, frames_per_clip: int) -> list[tuple[int, int]]:
  """Zero-based half-open ranges that retain every frame in <=limit clips.

  MotionLibrary needs at least two frames per logical clip. If division leaves a
  one-frame tail, one frame is moved from the preceding clip into that tail. Nothing
  is dropped and no range overlaps another.
  """
  if num_frames < 2:
    raise ValueError(f"A motion sequence needs at least 2 frames, got {num_frames}.")
  if frames_per_clip < 2:
    raise ValueError(f"frames_per_clip must be at least 2, got {frames_per_clip}.")
  if num_frames <= frames_per_clip:
    return [(0, num_frames)]

  ranges = [
    (start, min(start + frames_per_clip, num_frames))
    for start in range(0, num_frames, frames_per_clip)
  ]
  if ranges[-1][1] - ranges[-1][0] == 1:
    previous_start, previous_stop = ranges[-2]
    _, final_stop = ranges[-1]
    ranges[-2] = (previous_start, previous_stop - 1)
    ranges[-1] = (previous_stop - 1, final_stop)
  return ranges


def _resolved_motion_path(entry: dict[str, Any], manifest_dir: Path) -> Path:
  path = Path(entry["path"])
  return path if path.is_absolute() else (manifest_dir / path).resolve()


def _segment_entries(
  entries: list[dict[str, Any]],
  source_manifest_dir: Path,
  output_manifest_dir: Path,
  clip_seconds: float,
) -> list[dict[str, Any]]:
  """Create virtual 10-second clips without duplicating complete NPZ files."""
  if not np.isfinite(clip_seconds) or clip_seconds <= 0.0:
    raise ValueError("clip_seconds must be positive.")

  segmented: list[dict[str, Any]] = []
  for entry in entries:
    name = entry.get("name")
    if not isinstance(name, str) or not name:
      raise TypeError(f"Clip name must be a non-empty string, got {name!r}.")
    source = entry.get("source")
    if not isinstance(source, str) or not source:
      raise TypeError(f"{name}: source must be a non-empty string.")
    fps = float(entry["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
      raise ValueError(f"{name}: fps must be positive, got {fps}.")
    num_frames = entry.get("num_frames")
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
      raise TypeError(f"{name}: num_frames must be an integer, got {num_frames!r}.")
    base_start = entry.get("frame_start", 0)
    if isinstance(base_start, bool) or not isinstance(base_start, int):
      raise TypeError(f"{name}: frame_start must be an integer, got {base_start!r}.")
    if base_start < 0:
      raise ValueError(f"{name}: frame_start must be non-negative, got {base_start}.")
    # Floor rather than round so non-integral frame rates cannot exceed 10 seconds.
    frames_per_clip = int(clip_seconds * fps)
    ranges = _frame_ranges(num_frames, frames_per_clip)
    path = _resolved_motion_path(entry, source_manifest_dir)
    relative_path = os.path.relpath(path, output_manifest_dir.resolve())

    for index, (start, stop) in enumerate(ranges):
      clip = dict(entry)
      clip["name"] = name if len(ranges) == 1 else f"{name}__seg{index:03d}"
      clip["path"] = relative_path
      clip["frame_start"] = base_start + start
      clip["frame_stop"] = base_start + stop
      clip["num_frames"] = stop - start
      clip["sequence_name"] = name
      segmented.append(clip)

  names = [entry["name"] for entry in segmented]
  if len(names) != len(set(names)):
    raise ValueError("Stratification generated duplicate clip names.")
  return segmented


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  """Atomically write a JSON artifact."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(f"{path.suffix}.tmp")
  try:
    with temporary.open("w") as f:
      json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _relocate_entry(
  entry: dict[str, Any], from_dir: Path, to_dir: Path
) -> dict[str, Any]:
  relocated = dict(entry)
  relocated["path"] = os.path.relpath(
    _resolved_motion_path(entry, from_dir), to_dir.resolve()
  )
  return relocated


def _validate_strict_v1_config(cfg: Config) -> None:
  """Reject CLI overrides that would no longer implement Sec. IV-C v1."""
  if cfg.trials != STRATIFICATION_TRIALS:
    raise ValueError("Strict v1 stratification requires exactly 5 trials per clip.")
  if cfg.threshold != STRATIFICATION_THRESHOLD:
    raise ValueError("Strict v1 stratification requires threshold=0.8.")
  if cfg.clip_seconds != STRATIFICATION_MAX_CLIP_SECONDS:
    raise ValueError("Strict v1 stratification requires clip_seconds=10.0.")
  if cfg.dataset_label != "v1-proxy":
    raise ValueError(
      "Strict v1 uses dataset_label='v1-proxy'; do not claim unavailable AMASS/"
      "in-house Xsens inputs as exact paper data."
    )


def _artifact_references(
  owner: Path, artifact_paths: dict[str, Path]
) -> dict[str, str]:
  return {
    name: os.path.relpath(path.resolve(), owner.parent.resolve())
    for name, path in artifact_paths.items()
  }


def _artifact_payload(
  *,
  kind: str,
  owner: Path,
  artifact_paths: dict[str, Path],
  protocol: dict[str, Any],
  provenance: dict[str, Any],
  artifact_id: str,
  clips: list[dict[str, Any]] | dict[str, dict[str, Any]],
  **extra: Any,
) -> dict[str, Any]:
  return {
    "schema": STRATIFICATION_SCHEMA,
    "schema_version": STRATIFICATION_SCHEMA_VERSION,
    "kind": kind,
    "protocol": protocol,
    "provenance": provenance,
    "artifact_set_sha256": artifact_id,
    "artifacts": _artifact_references(owner, artifact_paths),
    **extra,
    "clips": clips,
  }


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
  _validate_strict_v1_config(cfg)
  source_manifest = Path(cfg.manifest)
  with source_manifest.open() as f:
    source_payload = json.load(f)
  if not isinstance(source_payload, dict):
    raise TypeError(f"{source_manifest} must contain a JSON object.")
  if source_payload.get("kind") != "complete_sequences":
    raise ValueError(
      f"{source_manifest} is not a complete-sequence Stage-I manifest. Regenerate "
      "it with prepare_motions before stratification."
    )
  source_entries = source_payload.get("clips")
  if not isinstance(source_entries, list) or not source_entries:
    raise ValueError(f"{source_manifest} must contain a non-empty clips list.")
  source_provenance = source_payload.get("provenance")
  if not isinstance(source_provenance, dict) or not source_provenance:
    raise ValueError(
      f"{source_manifest} must declare the proxy dataset provenance; strict v1 "
      "does not infer or invent AMASS/Xsens provenance."
    )

  stratified_manifest = Path(cfg.stratified_manifest)
  out_dir = Path(cfg.out_dir)
  artifact_paths = {
    "stratified": stratified_manifest,
    "mastered": out_dir / "mastered.json",
    "challenging": out_dir / "challenging.json",
    "report": out_dir / "stratification_report.json",
  }
  protocol = make_stratification_protocol(cfg.seed)
  provenance = make_stratification_provenance(
    source_manifest,
    cfg.checkpoint,
    dataset_label=cfg.dataset_label,
    input_manifest_provenance=source_provenance,
  )
  segmented = _segment_entries(
    source_entries,
    source_manifest.parent,
    stratified_manifest.parent,
    cfg.clip_seconds,
  )
  _write_json(
    stratified_manifest,
    _artifact_payload(
      kind="stratification_clips",
      owner=stratified_manifest,
      artifact_paths=artifact_paths,
      protocol=protocol,
      provenance=provenance,
      # This file is a transient MotionLibrary input. The complete authenticated
      # digest can only be computed after all five rollouts have finished.
      artifact_id="0" * 64,
      clips=segmented,
      provisional=True,
      source_manifest=os.path.relpath(
        source_manifest.resolve(), stratified_manifest.parent.resolve()
      ),
      clip_seconds=cfg.clip_seconds,
    ),
  )

  # Seed every RNG before environment construction. MuJoCo Warp discloses only
  # best-effort determinism, which is recorded verbatim in the protocol block.
  random_utils.seed_rng(cfg.seed)
  device = resolve_device(cfg.device)
  env, policy, command = build_env_and_policy(
    task_id=cfg.task,
    checkpoint=cfg.checkpoint,
    num_envs=cfg.num_envs,
    device=device,
    # Domain randomization stays on: that is what makes the rollouts "randomized".
    play=False,
    manifest=str(stratified_manifest),
  )
  try:
    lib = command.lib
    clip_ids = torch.arange(lib.num_clips, dtype=torch.long, device=device)

    print(f"[gmtrack] stratifying {lib.num_clips} clips x {cfg.trials} trials")
    results = rollout_clips(
      env, policy, command, clip_ids, rollouts_per_clip=cfg.trials
    )

    entries = {entry["name"]: entry for entry in segmented}
    annotated_entries: dict[str, dict[str, Any]] = {}
    mastered: list[dict[str, Any]] = []
    challenging: list[dict[str, Any]] = []
    report: dict[str, dict[str, Any]] = {}
    for clip_id, res in sorted(results.items()):
      entry = entries[res.name]
      classification = (
        "mastered" if res.completion_rate >= cfg.threshold else "challenging"
      )
      annotated_entry = {
        **entry,
        "trials": res.trials,
        "success_count": res.successes,
        "success_rate": res.completion_rate,
        "classification": classification,
      }
      annotated_entries[res.name] = annotated_entry
      target = mastered if classification == "mastered" else challenging
      target.append(
        _relocate_entry(annotated_entry, stratified_manifest.parent, out_dir)
      )
      report[res.name] = {
        **res.summary(),
        **_kinematic_stats(lib, clip_id),
        "trials": res.trials,
        "successes": res.successes,
        "success_count": res.successes,
        "success_rate": res.completion_rate,
        "classification": classification,
        "source": entry["source"],
        "sequence_name": entry["sequence_name"],
        "frame_start": entry["frame_start"],
        "frame_stop": entry["frame_stop"],
        "num_frames": entry["num_frames"],
        "fps": entry["fps"],
        "duration_seconds": entry["num_frames"] / entry["fps"],
      }

    subset_extra = {
      "stratified_manifest": os.path.relpath(
        stratified_manifest.resolve(), out_dir.resolve()
      )
    }
    final_segmented = [annotated_entries[entry["name"]] for entry in segmented]
    artifact_id = artifact_set_sha256(
      protocol,
      provenance,
      {
        "stratified": final_segmented,
        "mastered": mastered,
        "challenging": challenging,
        "report": report,
      },
    )
    _write_json(
      stratified_manifest,
      _artifact_payload(
        kind="stratification_clips",
        owner=stratified_manifest,
        artifact_paths=artifact_paths,
        protocol=protocol,
        provenance=provenance,
        artifact_id=artifact_id,
        clips=final_segmented,
        source_manifest=os.path.relpath(
          source_manifest.resolve(), stratified_manifest.parent.resolve()
        ),
        clip_seconds=cfg.clip_seconds,
      ),
    )
    _write_json(
      artifact_paths["mastered"],
      _artifact_payload(
        kind="mastered_clips",
        owner=artifact_paths["mastered"],
        artifact_paths=artifact_paths,
        protocol=protocol,
        provenance=provenance,
        artifact_id=artifact_id,
        clips=mastered,
        **subset_extra,
      ),
    )
    _write_json(
      artifact_paths["challenging"],
      _artifact_payload(
        kind="challenging_clips",
        owner=artifact_paths["challenging"],
        artifact_paths=artifact_paths,
        protocol=protocol,
        provenance=provenance,
        artifact_id=artifact_id,
        clips=challenging,
        **subset_extra,
      ),
    )
    _write_json(
      artifact_paths["report"],
      _artifact_payload(
        kind="stratification_report",
        owner=artifact_paths["report"],
        artifact_paths=artifact_paths,
        protocol=protocol,
        provenance=provenance,
        artifact_id=artifact_id,
        clips=report,
        **subset_extra,
      ),
    )

    # Validate the exact bytes that Stage II will consume, not only in-memory data.
    validate_stage2_manifests(
      stratified_manifest,
      artifact_paths["mastered"],
      artifact_paths["challenging"],
    )

    def hours(clips: list[dict[str, Any]]) -> float:
      return sum(c["num_frames"] / c["fps"] for c in clips) / 3600.0

    print(
      f"[gmtrack] mastered   D_m: {len(mastered):5d} clips, "
      f"{hours(mastered):.2f} h\n"
      f"[gmtrack] challenging D_c: {len(challenging):5d} clips, "
      f"{hours(challenging):.2f} h"
    )

    # Fig. 4: median of the per-motion 99th percentiles, per set.
    for label, clips in (("mastered", mastered), ("challenging", challenging)):
      if not clips:
        continue
      keys = list(_kinematic_stats(lib, 0).keys())
      medians = {
        key: float(np.median([report[clip["name"]][key] for clip in clips]))
        for key in keys
      }
      print(
        f"[gmtrack] {label:12s} "
        + "  ".join(f"{key}={value:.2f}" for key, value in medians.items())
      )
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
