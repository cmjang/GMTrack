"""Tests for the fail-closed motion ground-clearance audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ex_grmt.motion_grounding import DEFAULT_CLEARANCE, GroundClearanceReport
from ex_grmt.scripts.audit_ground_clearance import (
  Config,
  GroundClearanceAuditError,
  audit_manifest,
  main,
)


class _RootHeightGrounder:
  """Small deterministic stand-in; root z is the signed test clearance."""

  def measure(self, root_pos, root_quat_wxyz, joint_pos):
    del root_quat_wxyz, joint_pos
    distances = np.asarray(root_pos, dtype=np.float64)[:, 2]
    names = tuple(
      "left_knee_collision" if value == distances.min() else "left_foot_collision"
      for value in distances
    )
    return GroundClearanceReport(distances, names)


def _write_motion(path: Path, distances: list[float], **metadata) -> None:
  frames = len(distances)
  body_pos = np.zeros((frames, 30, 3), dtype=np.float32)
  body_pos[:, 0, 2] = distances
  body_quat = np.zeros((frames, 30, 4), dtype=np.float32)
  body_quat[:, :, 0] = 1.0
  arrays = {
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "joint_pos": np.zeros((frames, 29), dtype=np.float32),
    **{key: np.asarray([value]) for key, value in metadata.items()},
  }
  np.savez(path, **arrays)


def _write_manifest(
  path: Path,
  motion: Path,
  *,
  frame_start: int = 0,
  frame_stop: int | None = None,
) -> None:
  with np.load(motion) as data:
    stored_frames = int(data["joint_pos"].shape[0])
  stop = stored_frames if frame_stop is None else frame_stop
  path.write_text(
    json.dumps(
      {
        "clips": [
          {
            "name": "test-clip",
            "source": "unit-test",
            "path": motion.name,
            "frame_start": frame_start,
            "frame_stop": stop,
            "num_frames": stop - frame_start,
            "fps": 50.0,
          }
        ]
      }
    )
  )


def test_default_threshold_is_three_millimetres():
  assert Config(manifest="motion.json").threshold == DEFAULT_CLEARANCE == 0.003


def test_audit_resolves_relative_path_and_honours_logical_clip_window(tmp_path):
  motion = tmp_path / "motion.npz"
  manifest = tmp_path / "manifest.json"
  _write_motion(motion, [-0.02, 0.004, 0.002, -0.03])
  _write_manifest(manifest, motion, frame_start=1, frame_stop=3)

  report = audit_manifest(manifest, grounder=_RootHeightGrounder())

  clip = report["clips"][0]
  assert (
    report["manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
  )
  assert clip["frame_start"] == 1
  assert clip["frame_stop"] == 3
  assert clip["min_clearance_m"] == pytest.approx(0.002)
  assert clip["worst_frame"] == 2
  assert clip["worst_geom"] == "left_foot_collision"
  assert clip["penetrating_frame_count"] == 0
  assert clip["threshold_violation_frame_count"] == 1
  assert clip["threshold_violation_frame_ratio"] == 0.5
  assert report["summary"]["passed"] is False


def test_stored_collision_grounding_qc_is_checked_against_measurement(tmp_path):
  motion = tmp_path / "motion.npz"
  manifest = tmp_path / "manifest.json"
  _write_motion(
    motion,
    [0.003, 0.004],
    ground_alignment="g1_collision",
    ground_clearance_m=0.003,
    ground_smoothing_radius_s=0.3,
    ground_min_clearance_before_m=-0.01,
    ground_min_clearance_after_m=0.003,
    ground_max_correction_m=0.013,
    ground_affected_frame_ratio=0.5,
    ground_correction_vmax_mps=0.1,
    ground_correction_amax_mps2=1.0,
  )
  _write_manifest(manifest, motion)

  report = audit_manifest(manifest, grounder=_RootHeightGrounder())

  qc = report["clips"][0]["stored_ground_alignment_qc"]
  assert qc["present"] is True
  assert qc["valid"] is True
  assert qc["alignment"] == "g1_collision"
  assert qc["mismatches"] == []
  assert report["summary"]["metadata_checked_clip_count"] == 1
  assert report["summary"]["passed"] is True


def test_file_qc_uses_complete_npz_not_logical_clip_window(tmp_path):
  motion = tmp_path / "motion.npz"
  manifest = tmp_path / "manifest.json"
  _write_motion(
    motion,
    [0.003, 0.01],
    ground_alignment="g1_collision",
    ground_clearance_m=0.003,
    ground_smoothing_radius_s=0.3,
    ground_min_clearance_before_m=-0.01,
    ground_min_clearance_after_m=0.003,
    ground_max_correction_m=0.013,
    ground_affected_frame_ratio=0.5,
    ground_correction_vmax_mps=0.1,
    ground_correction_amax_mps2=1.0,
  )
  _write_manifest(manifest, motion, frame_start=1, frame_stop=2)

  report = audit_manifest(manifest, grounder=_RootHeightGrounder())

  clip = report["clips"][0]
  assert clip["min_clearance_m"] == pytest.approx(0.01)
  assert clip["stored_ground_alignment_qc"]["ground_min_clearance_after_m"] == 0.003
  assert clip["stored_ground_alignment_qc"]["valid"] is True


def test_partial_or_inconsistent_stored_qc_fails_closed(tmp_path):
  motion = tmp_path / "motion.npz"
  manifest = tmp_path / "manifest.json"
  _write_motion(motion, [0.01], ground_alignment="g1_collision")
  _write_manifest(manifest, motion)

  report = audit_manifest(manifest, grounder=_RootHeightGrounder())

  qc = report["clips"][0]["stored_ground_alignment_qc"]
  assert qc["present"] is True
  assert qc["valid"] is False
  assert "missing metadata keys" in qc["mismatches"][0]
  assert report["summary"]["metadata_mismatch_clip_count"] == 1
  assert report["summary"]["passed"] is False


def test_main_atomically_writes_failure_report_before_raising(tmp_path, monkeypatch):
  motion = tmp_path / "motion.npz"
  manifest = tmp_path / "manifest.json"
  output = tmp_path / "reports" / "audit.json"
  _write_motion(motion, [-0.01])
  _write_manifest(manifest, motion)
  monkeypatch.setattr(
    "ex_grmt.scripts.audit_ground_clearance.G1MotionGrounder",
    _RootHeightGrounder,
  )

  with pytest.raises(GroundClearanceAuditError, match="below threshold"):
    main(Config(manifest=str(manifest), out=str(output)))

  stored = json.loads(output.read_text())
  assert stored["summary"]["passed"] is False
  assert stored["summary"]["penetrating_frame_count"] == 1
  assert not list(output.parent.glob("*.tmp"))


def test_invalid_threshold_and_motion_shape_fail_fast(tmp_path):
  with pytest.raises(ValueError, match="threshold must be finite"):
    audit_manifest(tmp_path / "missing.json", threshold=float("nan"))

  motion = tmp_path / "motion.npz"
  manifest = tmp_path / "manifest.json"
  np.savez(
    motion,
    body_pos_w=np.zeros((1, 3)),
    body_quat_w=np.zeros((1, 30, 4)),
    joint_pos=np.zeros((1, 29)),
  )
  _write_manifest(manifest, motion)
  with pytest.raises(ValueError, match="body_pos_w must have shape"):
    audit_manifest(manifest, grounder=_RootHeightGrounder())
