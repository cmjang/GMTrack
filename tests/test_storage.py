"""Tests for STAR resampling and PACE role bookkeeping in the rollout storage."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from gmtrack.pace import pace_env_split
from gmtrack.rsl_rl.storage import STAR_GROUP, StarRolloutStorage

T, N, A = 6, 8, 3


def _storage(acquisition_fraction=None, **kw) -> StarRolloutStorage:
  obs = TensorDict(
    {"actor": torch.zeros(N, 4), STAR_GROUP: torch.zeros(N, 2)}, batch_size=[N]
  )
  return StarRolloutStorage(
    "rl", N, T, obs, [A], "cpu", acquisition_fraction=acquisition_fraction, **kw
  )


def _fill(
  storage: StarRolloutStorage,
  dones: torch.Tensor,
  star: torch.Tensor,
  tracking_failures: torch.Tensor | None = None,
):
  """Populate the buffers directly; add_transition needs a live distribution."""
  storage.dones.copy_(dones.view(T, N, 1).byte())
  storage.observations[STAR_GROUP].copy_(star.view(T, N, 2))
  if tracking_failures is None:
    tracking_failures = torch.zeros(T, N)
  storage.tracking_failures.copy_(tracking_failures.view(T, N, 1).bool())
  storage.tracking_failure_sources.fill_(1)
  storage.step = T


def test_env_split_matches_acquisition_fraction():
  s = _storage(acquisition_fraction=0.75)
  assert s.env_split == 6
  assert s.acq_flat_idx.numel() == T * 6
  assert s.con_flat_idx.numel() == T * 2
  # Flat layout is idx = t * num_envs + env.
  assert torch.all(s.acq_flat_idx % N < 6)
  assert torch.all(s.con_flat_idx % N >= 6)


def test_stage_one_has_no_consolidation_pool():
  s = _storage(acquisition_fraction=None)
  assert s.env_split == N
  assert s.con_flat_idx.numel() == 0


def test_fragment_ids_break_at_dones_and_never_cross_envs():
  s = _storage()
  dones = torch.zeros(T, N)
  dones[2, 0] = 1.0  # env 0 terminates after step 2
  _fill(s, dones, torch.zeros(T, N, 2))

  frags = s.fragment_ids().view(T, N)
  # env 0: steps 0..2 in one fragment, 3..5 in the next.
  assert frags[0, 0] == frags[1, 0] == frags[2, 0]
  assert frags[3, 0] == frags[4, 0] == frags[5, 0]
  assert frags[2, 0] != frags[3, 0]
  # An env with no dones stays a single fragment.
  assert len(torch.unique(frags[:, 1])) == 1
  # No fragment id is shared between environments.
  for a in range(N):
    for b in range(a + 1, N):
      assert not set(frags[:, a].tolist()) & set(frags[:, b].tolist())


def test_valid_sample_counts_stop_at_first_termination():
  s = _storage(acquisition_fraction=0.5)  # envs 0-3 acquisition, 4-7 consolidation
  dones = torch.zeros(T, N)
  dones[1, 0] = 1.0  # env 0 valid for steps 0,1 only
  dones[3, 5] = 1.0  # env 5 valid for steps 0..3
  _fill(s, dones, torch.zeros(T, N, 2), tracking_failures=dones)

  n_acq, n_con = s.valid_sample_counts()
  assert n_acq == 2 + T + T + T  # env0 truncated, envs 1-3 full
  assert n_con == T + 4 + T + T  # env5 truncated, envs 4,6,7 full


def test_valid_sample_counts_ignore_timeouts_and_motion_ends():
  s = _storage(acquisition_fraction=0.5)
  dones = torch.zeros(T, N)
  failures = torch.zeros(T, N)
  dones[1, 0] = 1.0  # timeout / ordinary reference end: not a tracking failure
  dones[2, 4] = 1.0
  dones[3, 1] = failures[3, 1] = 1.0  # a genuine tracking failure
  _fill(s, dones, torch.zeros(T, N, 2), tracking_failures=failures)

  assert s.valid_sample_counts() == (T + 4 + T + T, 4 * T)
  # Legacy behaviour ended the prefix on every combined done.
  assert s.valid_sample_counts("combined_done_prefix") == (
    2 + 4 + T + T,
    3 + T + T + T,
  )

  diagnostics = s.valid_sample_diagnostics()
  assert diagnostics["tracking_failure_acq_events"] == 1
  assert diagnostics["non_failure_done_acq_events"] == 1
  assert diagnostics["non_failure_done_con_events"] == 1
  assert diagnostics["valid_acq_gain_vs_combined_done"] == T - 2
  assert diagnostics["valid_con_gain_vs_combined_done"] == T - 3


def test_failure_prefix_requires_recorded_failure_masks():
  s = _storage(acquisition_fraction=0.5)
  s.step = T
  with pytest.raises(RuntimeError, match="Missing tracking failure masks"):
    s.valid_sample_counts()

  # The legacy diagnostic remains usable for old rollouts that only stored done.
  assert s.valid_sample_counts("combined_done_prefix") == (4 * T, 4 * T)


def test_record_tracking_failures_tracks_explicit_and_derived_sources():
  s = _storage(acquisition_fraction=0.5)
  s.record_tracking_failures(torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]))
  s.step += 1
  s.record_tracking_failures(torch.zeros(N), derived=True)

  assert bool(s.tracking_failures[0, 0])
  assert s.tracking_failure_sources[:2].tolist() == [1, 2]


def test_difficulty_normalization_splits_high_and_low():
  s = _storage()
  weights = torch.ones(T, N)
  weights[:, :4] = 2.0  # H
  weights[:, 4:] = 0.5  # E
  _fill(s, torch.zeros(T, N), torch.stack([weights, torch.zeros(T, N)], dim=-1))

  adv = torch.arange(T * N, dtype=torch.float32).view(T, N, 1)
  adv[:, :4] *= 100.0  # Make the two groups wildly different in scale.
  s.advantages.copy_(adv)
  s.normalize_advantages_by_difficulty()

  flat = s.advantages.view(T, N)
  high, low = flat[:, :4].flatten(), flat[:, 4:].flatten()
  # Each group is standardized on its own, so both end up ~zero-mean unit-std.
  assert abs(float(high.mean())) < 1e-4
  assert abs(float(low.mean())) < 1e-4
  assert abs(float(high.std()) - 1.0) < 0.1
  assert abs(float(low.std()) - 1.0) < 0.1


def test_difficulty_normalization_falls_back_when_high_group_is_empty():
  s = _storage()
  # All weights <= 1 -> H is empty, the split is meaningless.
  _fill(s, torch.zeros(T, N), torch.stack([torch.zeros(T, N)] * 2, dim=-1))
  s.advantages.copy_(torch.randn(T, N, 1) * 5.0 + 3.0)
  s.normalize_advantages_by_difficulty()
  flat = s.advantages.flatten()
  assert abs(float(flat.mean())) < 1e-4
  assert abs(float(flat.std()) - 1.0) < 0.1


def test_difficulty_normalization_ignores_consolidation_rows():
  s = _storage(acquisition_fraction=0.5)
  weights = torch.full((T, N), 0.5)
  weights[:, :2] = 2.0
  _fill(s, torch.zeros(T, N), torch.stack([weights, torch.zeros(T, N)], dim=-1))

  adv = torch.arange(T * N, dtype=torch.float32).view(T, N, 1)
  adv[:, :2] *= 100.0
  adv[:, 4:] = 1.0e9  # Consolidation must not affect either STAR group.
  s.advantages.copy_(adv)
  s.normalize_advantages_by_difficulty()

  normalized = s.advantages.squeeze(-1)
  assert abs(float(normalized[:, :2].std()) - 1.0) < 1e-4
  assert abs(float(normalized[:, 2:4].std()) - 1.0) < 1e-4
  assert torch.count_nonzero(normalized[:, 4:]) == 0


def test_plain_normalization_uses_acquisition_only():
  s = _storage(acquisition_fraction=0.5)
  _fill(s, torch.zeros(T, N), torch.zeros(T, N, 2))
  s.advantages[:, :4] = torch.arange(T * 4).view(T, 4, 1).float()
  s.advantages[:, 4:] = 1.0e9
  s.normalize_acquisition_advantages()

  normalized = s.advantages.squeeze(-1)
  assert abs(float(normalized[:, :4].mean())) < 1e-6
  assert abs(float(normalized[:, :4].std()) - 1.0) < 1e-6
  assert torch.count_nonzero(normalized[:, 4:]) == 0


def test_group_moments_all_reduce_sufficient_statistics(monkeypatch):
  s = _storage()
  values = torch.tensor([1.0, 3.0])

  def fake_all_reduce(stats, op):
    del op
    # A second rank contributes values [5, 7].
    stats += torch.tensor([2.0, 12.0, 74.0])

  monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
  monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
  monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

  count, mean, std = s._group_moments(
    values, torch.ones(2, dtype=torch.bool), distributed=True
  )
  assert count == 4
  assert float(mean) == pytest.approx(4.0)
  assert float(std) == pytest.approx(torch.tensor([1.0, 3.0, 5.0, 7.0]).std())


def test_star_pool_selects_high_advantage_fragments():
  s = _storage(acquisition_fraction=0.5, rho_topk=0.5)
  # Every acquisition transition is high difficulty and in bin 0.
  weights = torch.zeros(T, N)
  weights[:, :4] = 2.0
  bins = torch.zeros(T, N)
  _fill(s, torch.zeros(T, N), torch.stack([weights, bins], dim=-1))

  # env 0 has by far the best raw advantage; env 1-3 are poor.
  raw = torch.zeros(T, N, 1)
  raw[:, 0] = 10.0
  raw[:, 1:4] = -1.0
  s.raw_advantages.copy_(raw)

  pool, omega = s._build_star_pool()
  assert pool.numel() > 0
  assert omega.numel() == pool.numel()
  envs = (pool % N).unique().tolist()
  # rho_topk=0.5 over 4 candidate fragments keeps the best 2; env 0 must be one.
  assert 0 in envs
  # Consolidation environments are never eligible.
  assert all(e < 4 for e in envs)


def test_star_pool_applies_topk_independently_in_each_bin():
  s = _storage(acquisition_fraction=0.5, rho_topk=0.5)
  weights = torch.zeros(T, N)
  weights[:, :4] = 2.0
  bins = torch.zeros(T, N)
  bins[:, 2:4] = 1.0
  _fill(s, torch.zeros(T, N), torch.stack([weights, bins], dim=-1))

  # Bin 0 candidates: env 0 (100), env 1 (90). Bin 1: env 2 (2), env 3 (1).
  # A global top-2 would incorrectly select {0, 1}; per-bin top-k selects {0, 2}.
  raw = torch.zeros(T, N, 1)
  raw[:, 0] = 100.0
  raw[:, 1] = 90.0
  raw[:, 2] = 2.0
  raw[:, 3] = 1.0
  s.raw_advantages.copy_(raw)

  pool, _ = s._build_star_pool()
  assert set((pool % N).unique().tolist()) == {0, 2}


def test_star_pool_empty_without_high_difficulty_transitions():
  s = _storage(acquisition_fraction=0.5)
  _fill(s, torch.zeros(T, N), torch.zeros(T, N, 2))  # all weights 0 -> no H
  pool, omega = s._build_star_pool()
  assert pool.numel() == 0 and omega.numel() == 0


##
# PACE role split
##


def test_env_split_always_leaves_both_roles_populated():
  for num_envs in (2, 3, 5, 64, 4096):
    for xi in (0.05, 0.5, 0.8, 0.99):
      split = pace_env_split(xi, num_envs)
      assert 1 <= split <= num_envs - 1, (xi, num_envs, split)


def test_env_split_rejects_single_environment():
  """Regression: ``min(max(int(0.8*1), 1), 0)`` used to silently yield 0.

  That left the acquisition group empty, so every transition would have been
  optimized with the consolidation objective and no PPO update would occur -- with no
  error raised anywhere.
  """
  with pytest.raises(ValueError, match="at least 2 environments"):
    pace_env_split(0.8, 1)


def test_mini_batches_have_a_constant_size():
  """Regression: a lopsided pool used to produce one batch of size M+1.

  ``con_per_batch = max(M - acq_per_batch, 1)`` forced a consolidation row on top of
  a full acquisition batch, inflating that batch and skewing the gradient weight of
  the duplicated rows.
  """
  s = _storage(acquisition_fraction=0.99)  # 7 acq envs, 1 con env out of 8
  weights = torch.zeros(T, N)
  _fill(s, torch.zeros(T, N), torch.stack([weights, torch.zeros(T, N)], dim=-1))
  s.values.zero_()
  s.returns.zero_()
  s.actions_log_prob.zero_()
  s.distribution_params = (torch.zeros(T, N, A), torch.zeros(T, N, A))
  s.advantages.zero_()
  s.raw_advantages.zero_()

  # Chosen to actually reproduce the old failure: with M=4 and an acq share of
  # 42/48, `round(4 * 0.875) == 4 == M`, so the old `max(M - acq, 1)` appended a
  # 5th row. A coarser split rounds acq below M and hides the bug.
  num_mini_batches = 12
  expected = (T * N) // num_mini_batches
  assert expected == 4
  sizes = {
    b.observations.batch_size[0]
    for b in s.star_mini_batch_generator(num_mini_batches, num_epochs=2)
  }
  assert sizes == {expected}, sizes


def test_mini_batch_acq_mask_matches_row_layout():
  s = _storage(acquisition_fraction=0.5)
  _fill(s, torch.zeros(T, N), torch.zeros(T, N, 2))
  s.values.zero_()
  s.returns.zero_()
  s.actions_log_prob.zero_()
  s.distribution_params = (torch.zeros(T, N, A), torch.zeros(T, N, A))
  s.advantages.zero_()
  s.raw_advantages.zero_()

  for batch in s.star_mini_batch_generator(2, num_epochs=1):
    assert batch.acq_mask is not None
    # Acquisition rows come first, consolidation rows after; both groups non-empty.
    assert bool(batch.acq_mask[0]) and not bool(batch.acq_mask[-1])
    assert batch.acq_mask.numel() == batch.observations.batch_size[0]


def test_star_disabled_yields_empty_pool():
  s = _storage(acquisition_fraction=0.5, use_star=False)
  weights = torch.full((T, N), 2.0)
  _fill(s, torch.zeros(T, N), torch.stack([weights, torch.zeros(T, N)], dim=-1))
  s.raw_advantages.copy_(torch.randn(T, N, 1))
  assert s._build_star_pool()[0].numel() == 0
