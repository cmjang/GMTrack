"""Focused regression tests for the Extreme-RGMT adaptive sampler."""

from __future__ import annotations

import pytest
import torch

from ex_grmt.mdp.commands import _clamp_training_start_frame
from ex_grmt.mdp.sampling import AdaptiveBinSampler


def _sampler(**kwargs) -> AdaptiveBinSampler:
  params = dict(
    clip_ids=torch.tensor([1, 3]),
    clip_bins=torch.tensor([2, 1]),
    max_bins=2,
    num_library_clips=4,
    uniform_ratio=0.1,
    alpha=0.5,
    max_count_over_mean=10.0,
  )
  params.update(kwargs)
  return AdaptiveBinSampler(**params)


def test_sampled_training_starts_never_land_on_the_terminal_frame():
  local = torch.tensor([-3, 0, 8, 99])
  clip_lengths = torch.tensor([5, 2, 10, 4])
  assert _clamp_training_start_frame(local, clip_lengths).tolist() == [0, 0, 8, 2]


def test_kernel_size_one_matches_equations_12_and_13_exactly():
  sampler = _sampler(kernel_size=1)
  sampler.failed_ema.copy_(torch.tensor([[2.0, 1.0], [0.0, 0.0]]))
  sampler.refresh()

  # Normalize failures first, add eps_u / B to every valid bin, then normalize.
  expected = torch.tensor([[2 / 3 + 0.1 / 3, 1 / 3 + 0.1 / 3], [0.1 / 3, 0.0]])
  expected /= expected.sum()
  assert torch.allclose(sampler.probs, expected)


def test_c_max_is_relative_to_active_mean_and_ignores_padding():
  sampler = _sampler(kernel_size=1, max_count_over_mean=1.0)
  # The valid entries have mean 4, so Eq. 13 clips the first score from 9 to 4.
  # The large padded value must affect neither the mean nor the probabilities.
  sampler.failed_ema.copy_(torch.tensor([[9.0, 3.0], [0.0, 1_000.0]]))
  sampler.refresh()

  expected = torch.tensor([[4 / 7 + 0.1 / 3, 3 / 7 + 0.1 / 3], [0.1 / 3, 0.0]])
  expected /= expected.sum()
  assert torch.allclose(sampler.probs, expected)


def test_step_ema_all_reduces_pending_in_distributed_training(monkeypatch):
  sampler = _sampler()
  sampler.record_failures(torch.tensor([1]), torch.tensor([0]))
  calls = []

  monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
  monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
  monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

  def fake_all_reduce(tensor, op=None, group=None):
    calls.append((op, group))
    tensor.mul_(2)  # Both ranks recorded the same single failure.

  monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
  sampler.step_ema()

  assert calls == [(torch.distributed.ReduceOp.SUM, None)]
  assert sampler.failed_ema[0, 0].item() == pytest.approx(1.0)
  assert not sampler._pending.any()


def test_step_ema_can_keep_a_rank_local_curriculum(monkeypatch):
  sampler = _sampler()
  sampler.record_failures(torch.tensor([1]), torch.tensor([0]))

  def unexpected_all_reduce(*args, **kwargs):
    raise AssertionError("all_reduce should be disabled")

  monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_all_reduce)
  sampler.step_ema(synchronize_distributed=False)
  assert sampler.failed_ema[0, 0].item() == pytest.approx(0.5)


def test_state_dict_round_trip_restores_ema_pending_and_probabilities():
  source = _sampler()
  source.failed_ema.copy_(torch.tensor([[0.5, 1.5], [2.5, 0.0]]))
  source._pending.copy_(torch.tensor([[3.0, 2.0], [1.0, 0.0]]))
  source.refresh()
  state = source.state_dict()

  restored = _sampler()
  restored.load_state_dict(state)

  assert torch.equal(restored.failed_ema, source.failed_ema)
  assert torch.equal(restored._pending, source._pending)
  assert torch.equal(restored.probs, source.probs)
  # Saved tensors must not alias future mutations of the sampler.
  source.failed_ema.zero_()
  source._pending.zero_()
  assert restored.failed_ema.any()
  assert restored._pending.any()


def test_state_dict_rejects_a_different_sampler_topology():
  source = _sampler()
  different = AdaptiveBinSampler(
    clip_ids=torch.tensor([0, 3]),
    clip_bins=torch.tensor([2, 1]),
    max_bins=2,
    num_library_clips=4,
  )
  with pytest.raises(ValueError, match="clip_ids"):
    different.load_state_dict(source.state_dict())


def test_state_dict_rejects_changed_sampler_hyperparameters():
  source = _sampler()
  changed = _sampler(alpha=0.25)
  with pytest.raises(ValueError, match="alpha mismatch"):
    changed.load_state_dict(source.state_dict())
