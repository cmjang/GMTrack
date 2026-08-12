"""Import and collision-ground AMP_mjlab's recovery clip for Ex-GRMT.

AMP_mjlab uses its recovery asset as an AMP expert/reset-state pool, so the source
NPZ is not guaranteed to be a collision-consistent tracking reference.  This importer
keeps the source joint/root motion, raises the complete robot with a smooth upper
envelope against Ex-GRMT's active G1 collision geometry, refreshes translational body
velocities, and replaces every LAFAN ``fallAndGetUp`` clip in a Stage-I manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ex_grmt.motion_grounding import (
  DEFAULT_CLEARANCE,
  CorrectionSmoothing,
  G1MotionGrounder,
)

_REQUIRED_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_IMPORT_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("wb") as f:
      np.savez(f, **arrays)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  try:
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _scalar_fps(value: np.ndarray) -> float:
  values = np.asarray(value, dtype=np.float64).reshape(-1)
  if values.size != 1 or not np.isfinite(values[0]) or values[0] <= 0.0:
    raise ValueError(f"fps must contain one finite positive value, got {value!r}.")
  return float(values[0])


def _load_source(path: Path) -> tuple[dict[str, np.ndarray], float]:
  with np.load(path) as data:
    missing = [key for key in (*_REQUIRED_KEYS, "fps") if key not in data]
    if missing:
      raise KeyError(f"{path} is missing required arrays {missing}.")
    arrays = {key: np.asarray(data[key]).copy() for key in _REQUIRED_KEYS}
    fps = _scalar_fps(data["fps"])

  frames = arrays["joint_pos"].shape[0]
  expected_shapes = {
    "joint_pos": (frames, 29),
    "joint_vel": (frames, 29),
    "body_pos_w": (frames, 30, 3),
    "body_quat_w": (frames, 30, 4),
    "body_lin_vel_w": (frames, 30, 3),
    "body_ang_vel_w": (frames, 30, 3),
  }
  for key, expected in expected_shapes.items():
    if arrays[key].shape != expected:
      raise ValueError(
        f"{path}: {key} has shape {arrays[key].shape}, expected {expected}."
      )
    if not np.all(np.isfinite(arrays[key])):
      raise ValueError(f"{path}: {key} contains a non-finite value.")
  if frames < 2:
    raise ValueError(f"{path}: recovery clip must contain at least two frames.")
  return arrays, fps


def _ground_source(
  arrays: dict[str, np.ndarray],
  *,
  fps: float,
  clearance: float,
  smoothing_radius_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  result = G1MotionGrounder().ground(
    arrays["body_pos_w"][:, 0],
    arrays["body_quat_w"][:, 0],
    arrays["joint_pos"],
    clearance=clearance,
    smoothing=CorrectionSmoothing(
      output_fps=fps,
      smoothing_radius_s=smoothing_radius_s,
    ),
  )
  correction = np.asarray(result.correction, dtype=np.float64)
  correction_velocity = np.gradient(correction, 1.0 / fps)
  correction_acceleration = np.gradient(correction_velocity, 1.0 / fps)

  grounded = {key: value.copy() for key, value in arrays.items()}
  grounded["body_pos_w"][:, :, 2] += correction[:, None]
  grounded["body_lin_vel_w"][:, :, 2] += correction_velocity[:, None]

  required = np.asarray(result.required_correction, dtype=np.float64)
  min_distance = np.asarray(result.min_distance, dtype=np.float64)
  qc = {
    "ground_alignment": np.array(["g1_collision"]),
    "ground_clearance_m": np.array([clearance], dtype=np.float64),
    "ground_smoothing_radius_s": np.array([smoothing_radius_s], dtype=np.float64),
    "ground_min_clearance_before_m": np.array(
      [float(np.min(min_distance))], dtype=np.float64
    ),
    "ground_min_clearance_after_m": np.array(
      [float(np.min(min_distance + correction))], dtype=np.float64
    ),
    "ground_max_correction_m": np.array([float(np.max(correction))], dtype=np.float64),
    "ground_affected_frame_ratio": np.array(
      [float(np.mean(required > 0.0))], dtype=np.float64
    ),
    "ground_correction_vmax_mps": np.array(
      [float(np.max(np.abs(correction_velocity)))], dtype=np.float64
    ),
    "ground_correction_amax_mps2": np.array(
      [float(np.max(np.abs(correction_acceleration)))], dtype=np.float64
    ),
  }
  return grounded, qc


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--source",
    default=(
      "cankao/AMP_mjlab/src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz"
    ),
  )
  parser.add_argument(
    "--base-manifest",
    default=(
      "data/current/"
      "stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"
    ),
  )
  parser.add_argument(
    "--output-npz",
    default=(
      "data/datasets/stage1_full/amp_mjlab/"
      "amp_mjlab__fallAndGetUp1_subject1_grounded.npz"
    ),
  )
  parser.add_argument(
    "--output-manifest",
    required=True,
  )
  parser.add_argument(
    "--output-recovery-manifest",
    required=True,
  )
  parser.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE)
  parser.add_argument("--smoothing-radius-s", type=float, default=0.3)
  parser.add_argument(
    "--frame-start",
    type=int,
    default=0,
    help="First grounded AMP frame exposed to the training manifest.",
  )
  args = parser.parse_args()

  source = Path(args.source).resolve()
  base_manifest = Path(args.base_manifest).resolve()
  output_npz = Path(args.output_npz).resolve()
  output_manifest = Path(args.output_manifest).resolve()
  output_recovery_manifest = Path(args.output_recovery_manifest).resolve()
  if not source.is_file():
    raise FileNotFoundError(source)
  if not base_manifest.is_file():
    raise FileNotFoundError(base_manifest)
  if not np.isfinite(args.clearance) or args.clearance < 0.0:
    raise ValueError("clearance must be finite and non-negative.")
  if not np.isfinite(args.smoothing_radius_s) or args.smoothing_radius_s < 0.0:
    raise ValueError("smoothing-radius-s must be finite and non-negative.")

  arrays, fps = _load_source(source)
  source_frames = int(arrays["joint_pos"].shape[0])
  if not 0 <= args.frame_start < source_frames - 1:
    raise ValueError(
      f"frame-start must be in [0, {source_frames - 2}], got {args.frame_start}."
    )
  grounded, qc = _ground_source(
    arrays,
    fps=fps,
    clearance=args.clearance,
    smoothing_radius_s=args.smoothing_radius_s,
  )
  source_sha = _sha256(source)
  _write_npz(
    output_npz,
    {
      "amp_recovery_import_schema_version": np.array(
        [_IMPORT_SCHEMA_VERSION], dtype=np.int64
      ),
      "fps": np.array([fps], dtype=np.float64),
      "source_sha256": np.array([source_sha]),
      **qc,
      **grounded,
    },
  )

  base = json.loads(base_manifest.read_text())
  base_clips = base.get("clips")
  if not isinstance(base_clips, list) or not base_clips:
    raise ValueError(f"{base_manifest}: expected a non-empty clips list.")
  retained = [
    clip
    for clip in base_clips
    if "fallandgetup" not in str(clip.get("name", "")).casefold().replace("_", "")
  ]
  removed = [clip for clip in base_clips if clip not in retained]
  if not removed:
    raise ValueError(f"{base_manifest}: no fallAndGetUp clips were found to replace.")

  relative_npz = os.path.relpath(output_npz, output_manifest.parent)
  replacement = {
    "name": f"amp_mjlab__fallAndGetUp1_subject1_grounded__from_{args.frame_start}",
    "source": "amp_mjlab",
    "path": Path(relative_npz).as_posix(),
    "frame_start": args.frame_start,
    "frame_stop": source_frames,
    "num_frames": source_frames - args.frame_start,
    "fps": fps,
  }
  recovery_relative_npz = os.path.relpath(output_npz, output_recovery_manifest.parent)
  recovery_entry = {**replacement, "path": Path(recovery_relative_npz).as_posix()}
  _write_json(
    output_recovery_manifest,
    {
      "kind": "complete_sequences",
      "provenance": {
        "source": "AMP_mjlab Recovery/fallAndGetUp1_subject1.npz",
        "source_sha256": source_sha,
        "frame_start": args.frame_start,
        "ground_alignment": "g1_collision",
        "ground_clearance_m": args.clearance,
        "ground_smoothing_radius_s": args.smoothing_radius_s,
      },
      "clips": [recovery_entry],
    },
  )
  clips = [*retained, replacement]
  total_hours = (
    sum(int(clip["num_frames"]) / float(clip["fps"]) for clip in clips) / 3600.0
  )
  provenance = dict(base.get("provenance", {}))
  provenance["recovery_replacement"] = {
    "removed_lafan_fallandgetup_clips": len(removed),
    "removed_lafan_fallandgetup_frames": sum(
      int(clip["num_frames"]) for clip in removed
    ),
    "replacement_source": "AMP_mjlab Recovery/fallAndGetUp1_subject1.npz",
    "replacement_source_sha256": source_sha,
    "replacement_frame_start": args.frame_start,
    "replacement_frames": replacement["num_frames"],
    "ground_alignment": "g1_collision",
    "ground_clearance_m": args.clearance,
    "ground_smoothing_radius_s": args.smoothing_radius_s,
  }
  provenance["total_hours"] = round(total_hours, 4)
  _write_json(
    output_manifest,
    {
      "kind": base.get("kind", "complete_sequences"),
      "provenance": provenance,
      "clips": clips,
    },
  )

  print(
    f"wrote {output_npz} ({source_frames} stored frames, {fps:g} Hz); "
    f"min clearance {float(qc['ground_min_clearance_after_m'][0]):.6f} m"
  )
  print(f"wrote {output_recovery_manifest} (standalone recovery audit manifest)")
  print(
    f"wrote {output_manifest} ({len(clips)} clips, {total_hours:.4f} h); "
    f"replaced {len(removed)} LAFAN fallAndGetUp clips"
  )


if __name__ == "__main__":
  main()
