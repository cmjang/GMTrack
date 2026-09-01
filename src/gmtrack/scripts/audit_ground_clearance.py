"""Audit motion manifests against the active G1 collision geometry.

The audit is deliberately independent of the visual meshes.  Every logical clip is
replayed through :class:`gmtrack.motion_grounding.G1MotionGrounder`, including its
``frame_start`` / ``frame_stop`` window, and checked against a signed clearance from
the z=0 plane.  A report is emitted even when the clearance gate fails.

Usage::

    uv run python -m gmtrack.scripts.audit_ground_clearance \
        --manifest data/current/stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json \
        --out logs/eval/final_backflip_ground_clearance.json
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab
import numpy as np
import tyro

from gmtrack.motion_grounding import (
  DEFAULT_CLEARANCE,
  G1_JOINT_ORDER,
  G1MotionGrounder,
)

_SCHEMA = "gmtrack-ground-clearance-audit"
_SCHEMA_VERSION = 1
_GROUND_ALIGNMENT_VALUES = frozenset({"none", "g1_collision"})
_GROUND_METADATA_KEYS = (
  "ground_alignment",
  "ground_clearance_m",
  "ground_smoothing_radius_s",
  "ground_min_clearance_before_m",
  "ground_min_clearance_after_m",
  "ground_max_correction_m",
  "ground_affected_frame_ratio",
  "ground_correction_vmax_mps",
  "ground_correction_amax_mps2",
)


@dataclass
class Config:
  manifest: str
  """Motion manifest whose clip paths are resolved relative to the manifest."""
  threshold: float = DEFAULT_CLEARANCE
  """Required signed clearance in metres (3 mm by default)."""
  tolerance: float = 1.0e-6
  """Absolute numerical tolerance in metres for the fail-closed gate."""
  out: str | None = None
  """Optional JSON report path. Without it, the report is printed to stdout."""


class GroundClearanceAuditError(RuntimeError):
  """Raised after emitting a report that failed clearance or metadata checks."""


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as f:
    while chunk := f.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _validate_parameters(threshold: float, tolerance: float) -> None:
  if not np.isfinite(threshold) or threshold < 0.0:
    raise ValueError(f"threshold must be finite and non-negative, got {threshold}.")
  if not np.isfinite(tolerance) or tolerance < 0.0:
    raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}.")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
  with path.open() as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise TypeError(f"{path}: manifest root must be an object.")
  clips = payload.get("clips")
  if not isinstance(clips, list) or not clips:
    raise ValueError(f"{path}: manifest must contain a non-empty clips list.")
  if not all(isinstance(clip, dict) for clip in clips):
    raise TypeError(f"{path}: every clips entry must be an object.")
  return clips


def _integer(value: Any, name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer, got {value!r}.")
  return value


def _clip_window(
  entry: dict[str, Any], *, stored_frames: int, clip_name: str
) -> tuple[int, int]:
  start = _integer(entry.get("frame_start", 0), f"{clip_name}: frame_start")
  if "frame_stop" in entry:
    stop = _integer(entry["frame_stop"], f"{clip_name}: frame_stop")
  elif "num_frames" in entry:
    stop = start + _integer(entry["num_frames"], f"{clip_name}: num_frames")
  else:
    stop = stored_frames
  if not 0 <= start < stop <= stored_frames:
    raise ValueError(
      f"{clip_name}: invalid frame window [{start}, {stop}) for {stored_frames} "
      "stored frames."
    )
  if "num_frames" in entry:
    num_frames = _integer(entry["num_frames"], f"{clip_name}: num_frames")
    if num_frames != stop - start:
      raise ValueError(
        f"{clip_name}: num_frames={num_frames} does not match frame window "
        f"[{start}, {stop})."
      )
  return start, stop


def _motion_arrays(
  data: np.lib.npyio.NpzFile, *, clip_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  required = ("body_pos_w", "body_quat_w", "joint_pos")
  missing = [key for key in required if key not in data]
  if missing:
    raise ValueError(f"{clip_name}: motion NPZ is missing arrays {missing}.")
  body_pos = np.asarray(data["body_pos_w"])
  body_quat = np.asarray(data["body_quat_w"])
  joint_pos = np.asarray(data["joint_pos"])
  if body_pos.ndim != 3 or body_pos.shape[1] < 1 or body_pos.shape[2] != 3:
    raise ValueError(
      f"{clip_name}: body_pos_w must have shape (F, B, 3), got {body_pos.shape}."
    )
  if body_quat.ndim != 3 or body_quat.shape[1] < 1 or body_quat.shape[2] != 4:
    raise ValueError(
      f"{clip_name}: body_quat_w must have shape (F, B, 4), got {body_quat.shape}."
    )
  if joint_pos.ndim != 2 or joint_pos.shape[1] != len(G1_JOINT_ORDER):
    raise ValueError(
      f"{clip_name}: joint_pos must have shape (F, {len(G1_JOINT_ORDER)}), "
      f"got {joint_pos.shape}."
    )
  frame_counts = (body_pos.shape[0], body_quat.shape[0], joint_pos.shape[0])
  if len(set(frame_counts)) != 1:
    raise ValueError(
      f"{clip_name}: motion arrays have inconsistent frame counts {frame_counts}."
    )
  return body_pos[:, 0], body_quat[:, 0], joint_pos


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
  value = np.asarray(data[key])
  if value.size != 1:
    raise ValueError(f"{key} must contain exactly one scalar, got shape {value.shape}.")
  return value.reshape(-1)[0].item()


def _finite_metadata_float(data: np.lib.npyio.NpzFile, key: str) -> float:
  value = _scalar(data, key)
  if isinstance(value, bool):
    raise TypeError(f"{key} must be a numeric scalar, got {value!r}.")
  try:
    result = float(value)
  except (TypeError, ValueError) as exc:
    raise TypeError(f"{key} must be a numeric scalar, got {value!r}.") from exc
  if not np.isfinite(result):
    raise ValueError(f"{key} must be finite, got {result}.")
  return result


def _mismatch_if_not_close(
  mismatches: list[str],
  key: str,
  actual: float,
  expected: float,
  tolerance: float,
) -> None:
  if not np.isclose(actual, expected, rtol=1.0e-6, atol=tolerance):
    mismatches.append(f"{key}={actual:.12g}, expected {expected:.12g}")


def _stored_ground_qc(
  data: np.lib.npyio.NpzFile,
  *,
  measured_min: float,
  threshold: float,
  tolerance: float,
) -> dict[str, Any]:
  present = [key for key in _GROUND_METADATA_KEYS if key in data]
  if not present:
    return {"present": False, "valid": True, "mismatches": []}

  missing = [key for key in _GROUND_METADATA_KEYS if key not in data]
  if missing:
    return {
      "present": True,
      "valid": False,
      "mismatches": [f"missing metadata keys: {missing}"],
    }

  try:
    alignment = _scalar(data, "ground_alignment")
    if not isinstance(alignment, str):
      raise TypeError(f"ground_alignment must be a string scalar, got {alignment!r}.")
    values = {
      key: _finite_metadata_float(data, key)
      for key in _GROUND_METADATA_KEYS
      if key != "ground_alignment"
    }
  except (TypeError, ValueError) as exc:
    return {"present": True, "valid": False, "mismatches": [str(exc)]}

  mismatches: list[str] = []
  if alignment not in _GROUND_ALIGNMENT_VALUES:
    mismatches.append(
      f"ground_alignment={alignment!r}, expected one of "
      f"{sorted(_GROUND_ALIGNMENT_VALUES)}"
    )
  _mismatch_if_not_close(
    mismatches,
    "ground_clearance_m",
    values["ground_clearance_m"],
    threshold,
    tolerance,
  )
  _mismatch_if_not_close(
    mismatches,
    "ground_min_clearance_after_m",
    values["ground_min_clearance_after_m"],
    measured_min,
    tolerance,
  )

  nonnegative = (
    "ground_clearance_m",
    "ground_smoothing_radius_s",
    "ground_max_correction_m",
    "ground_correction_vmax_mps",
    "ground_correction_amax_mps2",
  )
  for key in nonnegative:
    if values[key] < 0.0:
      mismatches.append(f"{key} must be non-negative, got {values[key]:.12g}")
  affected_ratio = values["ground_affected_frame_ratio"]
  if not 0.0 <= affected_ratio <= 1.0:
    mismatches.append(
      f"ground_affected_frame_ratio must be within [0, 1], got {affected_ratio:.12g}"
    )

  if alignment == "none":
    _mismatch_if_not_close(
      mismatches,
      "ground_min_clearance_before_m",
      values["ground_min_clearance_before_m"],
      measured_min,
      tolerance,
    )
    for key in (
      "ground_max_correction_m",
      "ground_affected_frame_ratio",
      "ground_correction_vmax_mps",
      "ground_correction_amax_mps2",
    ):
      _mismatch_if_not_close(mismatches, key, values[key], 0.0, tolerance)
  elif alignment == "g1_collision":
    expected_max_correction = max(
      0.0,
      values["ground_clearance_m"] - values["ground_min_clearance_before_m"],
    )
    _mismatch_if_not_close(
      mismatches,
      "ground_max_correction_m",
      values["ground_max_correction_m"],
      expected_max_correction,
      tolerance,
    )
    if measured_min < threshold - tolerance:
      mismatches.append(
        "ground_alignment='g1_collision' but measured clearance is below the "
        "configured threshold"
      )

  return {
    "present": True,
    "valid": not mismatches,
    "alignment": alignment,
    **values,
    "mismatches": mismatches,
  }


def audit_manifest(
  manifest: str | Path,
  *,
  threshold: float = DEFAULT_CLEARANCE,
  tolerance: float = 1.0e-6,
  grounder: G1MotionGrounder | None = None,
) -> dict[str, Any]:
  """Return a deterministic clearance and stored-QC audit report."""
  _validate_parameters(threshold, tolerance)
  manifest_path = Path(manifest).resolve()
  entries = _load_manifest(manifest_path)
  checker = G1MotionGrounder() if grounder is None else grounder

  clip_reports: list[dict[str, Any]] = []
  # Stage-II manifests can contain several logical windows into one complete NPZ.
  # Measure each NPZ once, and compare its stored conversion QC to that complete
  # sequence rather than to an arbitrary logical window.
  measured_files: dict[Path, tuple[np.ndarray, tuple[str, ...], dict[str, Any]]] = {}
  total_frames = 0
  total_penetrating = 0
  total_violating = 0
  for index, entry in enumerate(entries):
    name = entry.get("name")
    if not isinstance(name, str) or not name:
      raise TypeError(f"clips[{index}].name must be a non-empty string.")
    stored_path = entry.get("path")
    if not isinstance(stored_path, str) or not stored_path:
      raise TypeError(f"{name}: path must be a non-empty string.")
    path = Path(stored_path)
    motion_path = path if path.is_absolute() else manifest_path.parent / path
    motion_path = motion_path.resolve()

    if motion_path not in measured_files:
      with np.load(motion_path) as data:
        root_pos, root_quat, joint_pos = _motion_arrays(data, clip_name=name)
        clearance = checker.measure(root_pos, root_quat, joint_pos)
        full_distances = np.asarray(clearance.min_distance, dtype=np.float64)
        if full_distances.shape != (root_pos.shape[0],):
          raise RuntimeError(
            f"{name}: grounder returned {full_distances.shape}, expected "
            f"{(root_pos.shape[0],)}."
          )
        full_geoms = tuple(clearance.worst_geom)
        if len(full_geoms) != root_pos.shape[0]:
          raise RuntimeError(
            f"{name}: grounder returned {len(full_geoms)} geom names for "
            f"{root_pos.shape[0]} frames."
          )
        stored_qc = _stored_ground_qc(
          data,
          measured_min=float(np.min(full_distances)),
          threshold=threshold,
          tolerance=tolerance,
        )
      measured_files[motion_path] = (full_distances, full_geoms, stored_qc)

    full_distances, full_geoms, stored_qc = measured_files[motion_path]
    start, stop = _clip_window(
      entry, stored_frames=full_distances.shape[0], clip_name=name
    )
    distances = full_distances[start:stop]
    geoms = full_geoms[start:stop]
    worst_local = int(np.argmin(distances))
    min_clearance = float(distances[worst_local])
    penetrating = distances < -tolerance
    violating = distances < threshold - tolerance

    num_frames = stop - start
    penetrating_count = int(np.count_nonzero(penetrating))
    violation_count = int(np.count_nonzero(violating))
    total_frames += num_frames
    total_penetrating += penetrating_count
    total_violating += violation_count
    clip_reports.append(
      {
        "name": name,
        "source": entry.get("source"),
        "path": stored_path,
        "frame_start": start,
        "frame_stop": stop,
        "num_frames": num_frames,
        "min_clearance_m": min_clearance,
        "worst_frame": start + worst_local,
        "worst_geom": geoms[worst_local],
        "penetrating_frame_count": penetrating_count,
        "penetrating_frame_ratio": penetrating_count / num_frames,
        "threshold_violation_frame_count": violation_count,
        "threshold_violation_frame_ratio": violation_count / num_frames,
        "passes_threshold": violation_count == 0,
        "stored_ground_alignment_qc": stored_qc,
      }
    )

  worst_clip = min(clip_reports, key=lambda clip: clip["min_clearance_m"])
  failing_clips = [clip for clip in clip_reports if not clip["passes_threshold"]]
  invalid_metadata = [
    clip for clip in clip_reports if not clip["stored_ground_alignment_qc"]["valid"]
  ]
  metadata_clips = [
    clip for clip in clip_reports if clip["stored_ground_alignment_qc"]["present"]
  ]
  passed = not failing_clips and not invalid_metadata
  return {
    "schema": _SCHEMA,
    "schema_version": _SCHEMA_VERSION,
    "manifest": {
      "path": str(manifest_path),
      "sha256": _sha256(manifest_path),
    },
    "parameters": {
      "threshold_m": threshold,
      "tolerance_m": tolerance,
      "penetration_definition": "signed_clearance_m < -tolerance_m",
      "threshold_violation_definition": (
        "signed_clearance_m < threshold_m - tolerance_m"
      ),
    },
    "clips": clip_reports,
    "summary": {
      "passed": passed,
      "clip_count": len(clip_reports),
      "frame_count": total_frames,
      "min_clearance_m": worst_clip["min_clearance_m"],
      "worst_clip": worst_clip["name"],
      "worst_geom": worst_clip["worst_geom"],
      "penetrating_frame_count": total_penetrating,
      "penetrating_frame_ratio": total_penetrating / total_frames,
      "threshold_violation_frame_count": total_violating,
      "threshold_violation_frame_ratio": total_violating / total_frames,
      "failing_clip_count": len(failing_clips),
      "metadata_checked_clip_count": len(metadata_clips),
      "metadata_mismatch_clip_count": len(invalid_metadata),
    },
  }


def _write_report(path: Path, report: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("w") as f:
      json.dump(report, f, indent=2, allow_nan=False)
      f.write("\n")
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def main(cfg: Config) -> dict[str, Any]:
  report = audit_manifest(
    cfg.manifest, threshold=cfg.threshold, tolerance=cfg.tolerance
  )
  if cfg.out is None:
    print(json.dumps(report, indent=2, allow_nan=False))
  else:
    _write_report(Path(cfg.out), report)
  if not report["summary"]["passed"]:
    summary = report["summary"]
    raise GroundClearanceAuditError(
      "Ground-clearance audit failed: "
      f"{summary['failing_clip_count']} clip(s) below threshold and "
      f"{summary['metadata_mismatch_clip_count']} clip(s) with invalid stored QC."
    )
  return report


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
