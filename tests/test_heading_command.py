"""Pure command/environment contracts for SONIC-style heading feedback."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from gmtrack.envs.env_cfg import (
  COMMAND_WINDOW_NOISE,
  COMMAND_WINDOW_RADIUS,
  command_window_noise,
  make_gmtrack_env_cfg,
)
from gmtrack.mdp.commands import relative_root_orientation_6d


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
  zeros = torch.zeros_like(angle)
  return quat_from_euler_xyz(zeros, zeros, angle)


def test_relative_root_orientation_6d_matches_sonic_channel_order():
  robot_yaw = torch.tensor([math.pi / 2, -math.pi / 2])
  ref_yaw = torch.tensor([math.pi, 0.0])

  actual = relative_root_orientation_6d(_yaw_quat(robot_yaw), _yaw_quat(ref_yaw))
  # Both pairs differ by +90 degrees. SONIC takes matrix[..., :2] and then flattens
  # row-major, so the values alternate between columns within each matrix row.
  expected = torch.tensor([[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]] * 2, dtype=actual.dtype)
  torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


def test_relative_root_orientation_is_closed_loop_but_world_yaw_invariant():
  robot_yaw = torch.tensor([0.2])
  ref_yaw = torch.tensor([0.9])
  baseline = relative_root_orientation_6d(_yaw_quat(robot_yaw), _yaw_quat(ref_yaw))

  # A common world-frame rotation must cancel from q_robot^-1 * q_ref.
  common_rotation = torch.tensor([1.1])
  rotated_together = relative_root_orientation_6d(
    _yaw_quat(robot_yaw + common_rotation),
    _yaw_quat(ref_yaw + common_rotation),
  )
  torch.testing.assert_close(rotated_together, baseline, atol=1e-6, rtol=0.0)

  # Rotating only the reference changes the Actor input, so accumulated heading
  # error is observable instead of being reduced to an open-loop yaw-rate command.
  ref_only = relative_root_orientation_6d(
    _yaw_quat(robot_yaw), _yaw_quat(ref_yaw + common_rotation)
  )
  assert not torch.allclose(ref_only, baseline)


def test_heading_noise_appends_six_channels_per_token_without_changing_baseline():
  num_tokens = 2 * COMMAND_WINDOW_RADIUS + 1
  assert len(COMMAND_WINDOW_NOISE) == num_tokens * 38
  assert command_window_noise(False) == COMMAND_WINDOW_NOISE

  heading_noise = command_window_noise(True)
  assert len(heading_noise) == num_tokens * 44
  for token_idx in range(num_tokens):
    token = heading_noise[token_idx * 44 : (token_idx + 1) * 44]
    assert token[:38] == COMMAND_WINDOW_NOISE[:38]
    assert token[38:] == (0.05,) * 6


def test_heading_cfg_adds_only_training_yaw_jitter_and_play_is_clean():
  baseline = make_gmtrack_env_cfg(manifest="unused.json")
  assert baseline.commands["motion"].heading_closed_loop is False
  assert baseline.commands["motion"].pose_range == {}
  assert (
    len(baseline.observations["command_window"].terms["window"].params["magnitude"])
    == 21 * 38
  )

  heading = make_gmtrack_env_cfg(manifest="unused.json", heading_closed_loop=True)
  assert heading.commands["motion"].heading_closed_loop is True
  assert heading.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
  assert (
    len(heading.observations["command_window"].terms["window"].params["magnitude"])
    == 21 * 44
  )

  play = make_gmtrack_env_cfg(
    manifest="unused.json", heading_closed_loop=True, play=True
  )
  assert play.commands["motion"].heading_closed_loop is True
  assert play.commands["motion"].pose_range == {}
  assert play.observations["command_window"].terms["window"].params["enabled"] is False
