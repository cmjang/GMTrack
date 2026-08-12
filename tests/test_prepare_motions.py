"""Tests for complete-sequence motion preparation."""

from __future__ import annotations

import json

import mjlab
import numpy as np
import pytest
import torch
import tyro
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.utils.lab_api.math import quat_slerp

from ex_grmt.motion_grounding import GroundClearanceReport, GroundingResult
from ex_grmt.scripts.prepare_motions import (
  MOTIONDECODE_HEADER,
  Config,
  _apply_ground_alignment,
  _ArrayMotionLoader,
  _batch_quat_slerp,
  _existing_num_frames,
  _expected_output_frames,
  _motion_csv_files,
  _output_identity,
  _read_source_motion,
  _validate_unique_stems,
  _write_npz,
)


def test_output_frame_count_matches_float32_arange_boundaries():
  # A pure ceil formula returns 249 here, while mjlab's float32 arange emits 250.
  assert _expected_output_frames(84, input_fps=10.0, output_fps=30.0) == 250


def test_documented_command_lines_actually_parse():
  """The invocations printed in the docstring / CLAUDE.md must work verbatim.

  mjlab's tyro configuration disables implicit boolean flags, so a bare ``--append``
  fails with "Missing value for argument". That is exactly the command a user copies
  when ingesting a second motion source, and the failure is at argument-parse time
  with no hint that the docs are wrong.
  """
  cfg = tyro.cli(
    Config,
    args=[
      "--input-dir",
      "data/datasets/raw/seed_backflip",
      "--source",
      "seed-backflip",
      "--input-fps",
      "120",
      "--input-format",
      "bones-seed",
      "--output-dir",
      "data/datasets/seed_backflip",
      "--manifest",
      "logs/data_build/manifests/seed_backflip.json",
    ],
    config=mjlab.TYRO_FLAGS,
  )
  assert cfg.append is False
  assert cfg.source == "seed-backflip"
  assert cfg.input_fps == 120.0
  assert cfg.input_format == "bones-seed"
  assert cfg.ground_alignment == "none"
  assert cfg.ground_clearance_m == pytest.approx(0.003)
  assert cfg.ground_smoothing_radius_s == pytest.approx(0.3)

  # The first-pass invocation, without --append.
  cfg = tyro.cli(
    Config,
    args=[
      "--input-dir",
      "data/datasets/raw/lafan1",
      "--source",
      "lafan1",
      "--input-format",
      "mjlab",
      "--input-fps",
      "30",
    ],
    config=mjlab.TYRO_FLAGS,
  )
  assert cfg.append is False
  assert cfg.replay_batch_frames == 500

  grounded_cfg = tyro.cli(
    Config,
    args=[
      "--input-dir",
      "data/datasets/raw/seed_backflip",
      "--source",
      "seed-backflip",
      "--input-format",
      "bones-seed",
      "--ground-alignment",
      "g1_collision",
      "--ground-clearance-m",
      "0.005",
      "--ground-smoothing-radius-s",
      "0.2",
    ],
    config=mjlab.TYRO_FLAGS,
  )
  assert grounded_cfg.ground_alignment == "g1_collision"
  assert grounded_cfg.ground_clearance_m == pytest.approx(0.005)
  assert grounded_cfg.ground_smoothing_radius_s == pytest.approx(0.2)


def test_bones_seed_units_are_scaled_before_velocity_computation():
  motion = np.zeros((4, 36), dtype=np.float32)
  motion[:, 2] = [75.0, 76.0, 77.0, 78.0]  # cm
  motion[:, 6] = 1.0  # identity in xyzw
  motion[:, 7] = [0.0, 90.0, 180.0, 270.0]  # degrees

  loader = _ArrayMotionLoader(
    motion,
    input_fps=120.0,
    output_fps=120.0,
    device="cpu",
    input_format="bones-seed",
  )

  torch.testing.assert_close(
    loader.motion_base_poss_input[:, 2], torch.tensor([0.75, 0.76, 0.77, 0.78])
  )
  torch.testing.assert_close(
    loader.motion_dof_poss_input[:, 0],
    torch.tensor([0.0, np.pi / 2, np.pi, 3 * np.pi / 2]),
  )
  torch.testing.assert_close(loader.motion_base_rots_input[:, 0], torch.ones(4))
  # One centimetre per 1/120 s is 1.2 m/s, proving scaling happened before diff.
  torch.testing.assert_close(
    loader.motion_base_lin_vels[:, 2], torch.full((3,), 1.2), atol=1e-5, rtol=1e-5
  )


def test_default_ground_alignment_is_a_noop():
  motion = np.zeros((5, 36), dtype=np.float32)
  motion[:, 2] = np.linspace(0.75, 0.79, 5)
  motion[:, 6] = 1.0
  loader = _ArrayMotionLoader(motion, 50.0, 50.0, "cpu", "mjlab")
  root_before = loader.motion_base_poss.clone()
  velocity_before = loader.motion_base_lin_vels.clone()
  cfg = Config(input_dir=".", source="test", input_format="mjlab")

  class StubGrounder:
    def measure(self, root_pos, root_quat_wxyz, joint_pos):
      assert root_pos.shape == (4, 3)
      assert root_quat_wxyz.shape == (4, 4)
      assert joint_pos.shape == (4, 29)
      return GroundClearanceReport(
        min_distance=np.array([0.01, 0.02, 0.03, 0.04]),
        worst_geom=("left_foot_collision",) * 4,
      )

  qc = _apply_ground_alignment(loader, cfg, StubGrounder())  # type: ignore[arg-type]

  torch.testing.assert_close(loader.motion_base_poss, root_before)
  torch.testing.assert_close(loader.motion_base_lin_vels, velocity_before)
  assert qc["ground_alignment"].item() == "none"
  assert qc["ground_min_clearance_before_m"].item() == pytest.approx(0.01)
  assert qc["ground_min_clearance_after_m"].item() == pytest.approx(0.01)
  assert qc["ground_max_correction_m"].item() == 0.0


def test_collision_grounding_recomputes_root_velocity_after_resampling():
  motion = np.zeros((5, 36), dtype=np.float32)
  motion[:, 2] = 0.75
  motion[:, 6] = 1.0
  loader = _ArrayMotionLoader(motion, 50.0, 50.0, "cpu", "mjlab")
  correction = np.array([0.0, 0.01, 0.03, 0.0], dtype=np.float64)

  class StubGrounder:
    def ground(
      self,
      root_pos,
      root_quat_wxyz,
      joint_pos,
      *,
      clearance,
      smoothing,
    ):
      assert root_pos.shape == (4, 3)
      assert root_quat_wxyz.shape == (4, 4)
      assert joint_pos.shape == (4, 29)
      assert clearance == pytest.approx(0.004)
      assert smoothing.output_fps == pytest.approx(50.0)
      assert smoothing.smoothing_radius_s == pytest.approx(0.2)
      corrected = root_pos.astype(np.float64, copy=True)
      corrected[:, 2] += correction
      min_distance = 0.004 - correction
      return GroundingResult(
        root_pos=corrected,
        correction=correction,
        required_correction=correction,
        min_distance=min_distance,
        worst_geom=("left_foot_collision",) * 4,
      )

  cfg = Config(
    input_dir=".",
    source="test",
    input_format="mjlab",
    output_fps=50.0,
    ground_alignment="g1_collision",
    ground_clearance_m=0.004,
    ground_smoothing_radius_s=0.2,
  )
  qc = _apply_ground_alignment(loader, cfg, StubGrounder())  # type: ignore[arg-type]

  expected_z = torch.tensor([0.75, 0.76, 0.78, 0.75])
  torch.testing.assert_close(loader.motion_base_poss[:, 2], expected_z)
  expected_velocity = torch.gradient(expected_z, spacing=1.0 / 50.0)[0]
  torch.testing.assert_close(loader.motion_base_lin_vels[:, 2], expected_velocity)
  assert qc["ground_min_clearance_before_m"].item() == pytest.approx(-0.026)
  assert qc["ground_min_clearance_after_m"].item() == pytest.approx(0.004)
  assert qc["ground_max_correction_m"].item() == pytest.approx(0.03)
  assert qc["ground_affected_frame_ratio"].item() == pytest.approx(0.5)
  assert qc["ground_correction_vmax_mps"].item() > 0.0
  assert qc["ground_correction_amax_mps2"].item() > 0.0


def test_motiondecode_loader_preserves_wxyz_quaternion_order():
  motion = np.zeros((4, 36), dtype=np.float32)
  motion[:, 2] = 0.8
  motion[:, 3:7] = np.array([2**-0.5, 2**-0.5, 0.0, 0.0])

  loader = _ArrayMotionLoader(
    motion,
    input_fps=120.0,
    output_fps=120.0,
    device="cpu",
    input_format="motiondecode",
  )

  torch.testing.assert_close(
    loader.motion_base_rots_input, torch.from_numpy(motion[:, 3:7])
  )


def test_batch_slerp_matches_mjlab_scalar_implementation():
  generator = torch.Generator().manual_seed(7)
  a = torch.nn.functional.normalize(torch.randn(64, 4, generator=generator), dim=1)
  b = torch.nn.functional.normalize(torch.randn(64, 4, generator=generator), dim=1)
  blend = torch.linspace(0.0, 1.0, 64)

  expected = torch.stack(
    [quat_slerp(a[i], b[i].clone(), float(blend[i])) for i in range(a.shape[0])]
  )
  torch.testing.assert_close(_batch_quat_slerp(a, b, blend), expected)


def test_array_loader_preserves_mjlab_resampling_semantics(tmp_path):
  num_frames = 1201
  phase = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
  motion = np.zeros((num_frames, 36), dtype=np.float32)
  motion[:, :3] = np.stack([phase, phase**2, 0.8 + 0.1 * phase], axis=1)
  angle = phase * np.pi
  motion[:, 5] = np.sin(angle / 2.0)
  motion[:, 6] = np.cos(angle / 2.0)
  motion[:, 7:] = phase[:, None] * np.linspace(-1.0, 1.0, 29)
  path = tmp_path / "motion.csv"
  np.savetxt(path, motion, delimiter=",")

  reference = MotionLoader(str(path), 120, 50, "cpu")
  actual = _ArrayMotionLoader(motion, 120, 50, "cpu", "mjlab")
  for attribute in (
    "motion_base_poss",
    "motion_base_rots",
    "motion_dof_poss",
    "motion_base_lin_vels",
    "motion_base_ang_vels",
    "motion_dof_vels",
  ):
    torch.testing.assert_close(
      getattr(actual, attribute), getattr(reference, attribute)
    )


def test_motion_csv_discovery_excludes_filtered_metadata(tmp_path):
  motion = tmp_path / "flip_360.csv"
  metadata = tmp_path / "filtered_metadata.csv"
  motion.touch()
  metadata.touch()
  assert _motion_csv_files(tmp_path) == [motion]


def test_motiondecode_header_selection_and_nested_output_identity(tmp_path):
  source = tmp_path / "samples" / "walk" / "same.csv"
  source.parent.mkdir(parents=True)
  motion = np.zeros((4, 36), dtype=np.float32)
  motion[:, 2] = 0.8
  motion[:, 3] = 1.0
  with source.open("w") as f:
    f.write(",".join(MOTIONDECODE_HEADER) + "\n")
    np.savetxt(f, motion, delimiter=",")
  selection = tmp_path / "selection.json"
  selection.write_text(json.dumps({"selected_files": ["samples/walk/same.csv"]}))

  np.testing.assert_array_equal(_read_source_motion(source, "motiondecode"), motion)
  assert _motion_csv_files(tmp_path, selection) == [source]
  name, output = _output_identity(
    source, tmp_path, tmp_path / "npz", "motiondecode", preserve=True
  )
  assert name == "motiondecode__samples__walk__same"
  assert output == tmp_path / "npz" / "samples" / "walk" / "same.npz"


def test_motiondecode_rejects_incompatible_header(tmp_path):
  source = tmp_path / "bad.csv"
  source.write_text("x,y\n1,2\n")
  with pytest.raises(ValueError, match="incompatible MotionDecode header"):
    _read_source_motion(source, "motiondecode")


def test_duplicate_stems_are_rejected_before_conversion(tmp_path):
  first = tmp_path / "a" / "same.csv"
  second = tmp_path / "b" / "same.csv"
  first.parent.mkdir()
  second.parent.mkdir()
  first.touch()
  second.touch()
  with pytest.raises(ValueError, match="duplicate CSV stem"):
    _validate_unique_stems([first, second])


def test_npz_write_is_atomic_when_serialization_fails(tmp_path, monkeypatch):
  destination = tmp_path / "clip.npz"
  original = b"previous-valid-output"
  destination.write_bytes(original)

  def fail_after_partial_write(file, **arrays):
    del arrays
    file.write(b"truncated")
    raise RuntimeError("injected failure")

  monkeypatch.setattr(np, "savez", fail_after_partial_write)
  with pytest.raises(RuntimeError, match="injected failure"):
    _write_npz(destination, fps=np.array([50.0]))
  assert destination.read_bytes() == original
  assert not list(tmp_path.glob(".*.tmp"))


def test_resume_rejects_changed_conversion_fingerprint(tmp_path):
  source = tmp_path / "motion.csv"
  source.write_text("source")
  source_stat = source.stat()
  output = tmp_path / "motion.npz"
  cfg = Config(
    input_dir=str(tmp_path), source="test", input_format="mjlab", input_fps=30.0
  )
  num_frames = 49
  arrays = {
    "converter_schema_version": np.array([4], dtype=np.int64),
    "fps": np.array([50.0]),
    "input_fps": np.array([30.0]),
    "input_format": np.array(["mjlab"]),
    "ground_alignment": np.array(["none"]),
    "ground_clearance_m": np.array([0.003]),
    "ground_smoothing_radius_s": np.array([0.3]),
    "ground_min_clearance_before_m": np.array([0.01]),
    "ground_min_clearance_after_m": np.array([0.01]),
    "ground_max_correction_m": np.array([0.0]),
    "ground_affected_frame_ratio": np.array([0.0]),
    "ground_correction_vmax_mps": np.array([0.0]),
    "ground_correction_amax_mps2": np.array([0.0]),
    "line_start": np.array([1], dtype=np.int64),
    "line_stop": np.array([30], dtype=np.int64),
    "source_size": np.array([source_stat.st_size], dtype=np.int64),
    "source_mtime_ns": np.array([source_stat.st_mtime_ns], dtype=np.int64),
    "joint_pos": np.zeros((num_frames, 29)),
    "joint_vel": np.zeros((num_frames, 29)),
    "body_pos_w": np.zeros((num_frames, 30, 3)),
    "body_quat_w": np.zeros((num_frames, 30, 4)),
    "body_lin_vel_w": np.zeros((num_frames, 30, 3)),
    "body_ang_vel_w": np.zeros((num_frames, 30, 3)),
  }
  _write_npz(output, **arrays)
  assert _existing_num_frames(output, cfg, (1, 30), source_stat) == num_frames

  changed = Config(
    input_dir=str(tmp_path),
    source="test",
    input_format="mjlab",
    input_fps=30.0,
    output_fps=60.0,
  )
  with pytest.raises(ValueError, match="has fps=50.0, expected 60.0"):
    _existing_num_frames(output, changed, (1, 30), source_stat)

  grounded = Config(
    input_dir=str(tmp_path),
    source="test",
    input_format="mjlab",
    input_fps=30.0,
    ground_alignment="g1_collision",
  )
  with pytest.raises(ValueError, match="has ground_alignment='none'"):
    _existing_num_frames(output, grounded, (1, 30), source_stat)

  changed_clearance = Config(
    input_dir=str(tmp_path),
    source="test",
    input_format="mjlab",
    input_fps=30.0,
    ground_clearance_m=0.004,
  )
  with pytest.raises(ValueError, match="has ground_clearance_m=0.003"):
    _existing_num_frames(output, changed_clearance, (1, 30), source_stat)


def test_bones_seed_rejects_non_120_fps(tmp_path):
  from ex_grmt.scripts.prepare_motions import main

  with pytest.raises(ValueError, match="input-fps 120"):
    main(
      Config(
        input_dir=str(tmp_path),
        source="seed",
        input_fps=30.0,
        input_format="bones-seed",
      )
    )


def test_motiondecode_rejects_non_120_fps(tmp_path):
  from ex_grmt.scripts.prepare_motions import main

  with pytest.raises(ValueError, match="MotionDecode.*input-fps 120"):
    main(
      Config(
        input_dir=str(tmp_path),
        source="motiondecode",
        input_fps=30.0,
        input_format="motiondecode",
      )
    )
