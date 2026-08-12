"""Tests for collision-geometry-based G1 motion grounding."""

from __future__ import annotations

import numpy as np
import pytest

from ex_grmt.motion_grounding import (
  G1_FK_BODY_ORDER,
  G1_JOINT_ORDER,
  CorrectionSmoothing,
  G1MotionGrounder,
  required_correction,
  smooth_correction_upper_envelope,
)


@pytest.fixture(scope="module")
def grounder() -> G1MotionGrounder:
  return G1MotionGrounder()


def _standing(num_frames: int = 1):
  root_pos = np.tile((1.25, -0.75, 1.0), (num_frames, 1))
  root_quat = np.tile((1.0, 0.0, 0.0, 0.0), (num_frames, 1))
  joint_pos = np.zeros((num_frames, 29))
  return root_pos, root_quat, joint_pos


def test_fk_layout_is_the_motion_file_depth_first_order(grounder):
  model = grounder.model
  assert tuple(model.body(i).name for i in range(1, model.nbody)) == G1_FK_BODY_ORDER
  assert tuple(model.joint(i).name for i in range(1, model.njnt)) == G1_JOINT_ORDER


def test_measure_uses_signed_collision_distance_and_tracks_vertical_translation(
  grounder,
):
  root_pos, root_quat, joint_pos = _standing(2)
  root_pos[1, 2] += 0.375

  report = grounder.measure(root_pos, root_quat, joint_pos)

  assert report.min_distance[0] == pytest.approx(0.20813585301208443)
  assert report.min_distance[1] - report.min_distance[0] == pytest.approx(0.375)
  assert report.worst_geom == (
    "left_foot4_collision",
    "left_foot4_collision",
  )


def test_ground_applies_exact_upward_only_correction(grounder):
  root_pos, root_quat, joint_pos = _standing(2)
  root_pos[:, 2] = (0.7, 1.2)
  clearance = 0.015

  result = grounder.ground(
    root_pos,
    root_quat * 2.0,
    joint_pos,
    clearance=clearance,
    smoothing=None,
  )

  expected = required_correction(result.min_distance, clearance=clearance)
  np.testing.assert_allclose(result.required_correction, expected, rtol=0.0, atol=0.0)
  np.testing.assert_allclose(result.correction, expected, rtol=0.0, atol=0.0)
  np.testing.assert_allclose(result.root_pos[:, :2], root_pos[:, :2])
  np.testing.assert_allclose(result.root_pos[:, 2], root_pos[:, 2] + result.correction)
  assert result.correction[0] > 0.0
  assert result.correction[1] == 0.0

  corrected = grounder.measure(result.root_pos, root_quat, joint_pos)
  assert corrected.min_distance[0] == pytest.approx(clearance, abs=1e-12)
  assert corrected.min_distance[1] >= clearance


def test_smoothed_upper_envelope_never_undercorrects(grounder):
  root_pos, root_quat, joint_pos = _standing(9)
  root_pos[4, 2] = 0.7
  smoothing = CorrectionSmoothing(
    output_fps=50.0,
    smoothing_radius_s=0.04,
    gaussian_sigma_s=0.02,
  )

  result = grounder.ground(
    root_pos, root_quat, joint_pos, clearance=0.003, smoothing=smoothing
  )

  assert np.all(result.correction >= result.required_correction)
  assert result.correction[3] > result.required_correction[3]
  assert result.correction[5] > result.required_correction[5]
  corrected = grounder.measure(result.root_pos, root_quat, joint_pos)
  assert np.all(corrected.min_distance >= 0.003 - 1e-12)


def test_smoothing_function_keeps_exact_required_peaks():
  required = np.array([0.0, 0.0, 0.2, 0.0, 0.0])
  smoothed = smooth_correction_upper_envelope(
    required,
    output_fps=10.0,
    smoothing_radius_s=0.1,
    gaussian_sigma_s=0.05,
  )
  assert np.all(smoothed >= required)
  assert smoothed[2] == required[2]
  assert smoothed[1] > 0.0
  assert smoothed[3] > 0.0


@pytest.mark.parametrize(
  ("argument", "value", "match"),
  [
    ("root_pos", np.zeros((2, 2)), r"root_pos must have shape \(F, 3\)"),
    (
      "root_quat",
      np.zeros((2, 3)),
      r"root_quat_wxyz must have shape \(F, 4\)",
    ),
    ("joint_pos", np.zeros((2, 28)), r"joint_pos must have shape \(F, 29\)"),
  ],
)
def test_shape_errors_fail_fast(grounder, argument, value, match):
  root_pos, root_quat, joint_pos = _standing(2)
  inputs = {
    "root_pos": root_pos,
    "root_quat": root_quat,
    "joint_pos": joint_pos,
  }
  inputs[argument] = value
  with pytest.raises(ValueError, match=match):
    grounder.measure(inputs["root_pos"], inputs["root_quat"], inputs["joint_pos"])


def test_frame_count_mismatch_fails_fast(grounder):
  root_pos, root_quat, joint_pos = _standing(2)
  with pytest.raises(ValueError, match="must have the same frame count"):
    grounder.measure(root_pos, root_quat[:1], joint_pos)


@pytest.mark.parametrize("argument", ["root_pos", "root_quat", "joint_pos"])
def test_nonfinite_input_fails_fast(grounder, argument):
  root_pos, root_quat, joint_pos = _standing()
  inputs = {
    "root_pos": root_pos,
    "root_quat": root_quat,
    "joint_pos": joint_pos,
  }
  inputs[argument] = inputs[argument].copy()
  inputs[argument].flat[0] = np.nan
  with pytest.raises(ValueError, match="non-finite"):
    grounder.measure(inputs["root_pos"], inputs["root_quat"], inputs["joint_pos"])


def test_invalid_quaternion_and_clearance_fail_fast(grounder):
  root_pos, root_quat, joint_pos = _standing()
  root_quat[:] = 0.0
  with pytest.raises(ValueError, match="zero-length quaternion"):
    grounder.measure(root_pos, root_quat, joint_pos)
  with pytest.raises(ValueError, match="finite and non-negative"):
    grounder.ground(
      root_pos, np.array([[1.0, 0.0, 0.0, 0.0]]), joint_pos, clearance=-0.1
    )
