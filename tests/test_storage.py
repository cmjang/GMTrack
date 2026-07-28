"""Tests for STAR resampling and PACE role bookkeeping in the rollout storage."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from ex_grmt.pace import pace_env_split
from ex_grmt.rsl_rl.storage import STAR_GROUP, StarRolloutStorage

T, N, A = 6, 8, 3


def _storage(acquisition_fraction=None, **kw) -> StarRolloutStorage:
  obs = TensorDict(
    {"actor": torch.zeros(N, 4), STAR_GROUP: torch.zeros(N, 2)}, batch_size=[N]
  )
  return StarRolloutStorage(
    "rl", N, T, obs, [A], "cpu", acquisition_fraction=acquisition_fraction, **kw
  )


def _fill(storage: StarRolloutStorage, dones: torch.Tensor, star: torch.Tensor):
  """Populate the buffers directly; add_transition needs a live distribution."""
  storage.dones.copy_(dones.view(T, N, 1).byte())
  storage.observations[STAR_GROUP].copy_(star.view(T, N, 2))
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
  _fill(s, dones, torch.zeros(T, N, 2))

  n_acq, n_con = s.valid_sample_counts()
  assert n_acq == 2 + T + T + T  # env0 truncated, envs 1-3 full
  assert n_con == T + 4 + T + T  # env5 truncated, envs 4,6,7 full


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
