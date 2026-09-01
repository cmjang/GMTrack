from __future__ import annotations

from pathlib import Path

import pytest
from torch.utils.tensorboard import SummaryWriter

from gmtrack.scripts.check_stage2_gate import Config, check_gate


def _write_gate(path: Path, *, bad_floor_steps: int) -> None:
  writer = SummaryWriter(path)
  for step in range(120):
    high_kl = step >= 120 - bad_floor_steps
    writer.add_scalar("Loss/kl", 0.03 if high_kl else 0.01, step)
    writer.add_scalar("Loss/learning_rate", 1e-5 if high_kl else 1e-4, step)
    writer.add_scalar("Loss/actor_grad_norm", 1.0, step)
    writer.add_scalar("Loss/critic_grad_norm", 1.0, step)
    writer.add_scalar("Loss/initial_deterministic_mean_max_abs_diff", 0.0, step)
    writer.add_scalar("Loss/initial_reference_kl_max", 0.0, step)
    writer.add_scalar("Loss/tracking_failure_explicit_steps", 24.0, step)
    writer.add_scalar("Loss/tracking_failure_derived_steps", 0.0, step)
    writer.add_scalar("Loss/tracking_failure_missing_steps", 0.0, step)
    writer.add_scalar("Episode_Termination/nonfinite_physics_state", 0.0, step)
  writer.close()


def test_gate_accepts_finite_trust_region_run(tmp_path: Path):
  _write_gate(tmp_path, bad_floor_steps=0)
  result = check_gate(Config(str(tmp_path), expected_iterations=120))
  assert result["passed"] is True


def test_gate_rejects_persistent_high_kl_at_lr_floor(tmp_path: Path):
  _write_gate(tmp_path, bad_floor_steps=100)
  result = check_gate(Config(str(tmp_path), expected_iterations=120))
  assert result["passed"] is False
  assert result["observed"]["floor_high_kl_streak"] == 100


def test_gate_rejects_incomplete_run(tmp_path: Path):
  _write_gate(tmp_path, bad_floor_steps=0)
  with pytest.raises(ValueError, match="incomplete"):
    check_gate(Config(str(tmp_path), expected_iterations=500))


def test_gate_rejects_compatibility_derived_tracking_failures(tmp_path: Path):
  _write_gate(tmp_path, bad_floor_steps=0)
  writer = SummaryWriter(tmp_path)
  for step in range(120):
    writer.add_scalar("Loss/tracking_failure_explicit_steps", 0.0, step)
    writer.add_scalar("Loss/tracking_failure_derived_steps", 24.0, step)
  writer.close()

  result = check_gate(Config(str(tmp_path), expected_iterations=120))
  assert result["passed"] is False
  assert result["observed"]["tracking_failure_hook_ok"] is False
