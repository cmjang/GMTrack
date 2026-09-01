"""Rollout storage with STAR resampling and PACE environment roles.

STAR (paper Sec. V-B) reshapes how the acquisition-side PPO mini-batches are drawn:

1. Every transition carries a difficulty weight ``w_t = B * p_{b_t}`` (Eq. 20) and the
   id of its reference bin, delivered through the ``star`` observation group.
2. Advantages are normalized **separately** inside the high-difficulty group
   ``H = {w > 1}`` and the remainder ``E = {w <= 1}`` (Eq. 23), so a handful of huge
   advantages in easy regions cannot flatten the signal in hard ones.
3. Contiguous trajectory fragments are scored by their mean *raw* advantage
   ``q_{b,tau}`` (Eq. 25); the top ``rho_topk`` per difficulty bin form a resampling
   pool (Eq. 26-27).
4. A fraction ``rho_star`` of each acquisition mini-batch is drawn from that pool,
   weighted by fragment score ``eta_tau`` (Eq. 28-30).

PACE (Sec. V-A) additionally partitions environments: indices ``[0, xi*N)`` are
acquisition environments, the rest are consolidation environments. Because the split
is by environment index and the flattened layout is ``idx = t * num_envs + env``, the
role of any flat index is simply ``idx % num_envs < split``.

Deviation from upstream worth knowing: rsl-rl 5.4.0 draws its shuffle **once**
outside the epoch loop, so all ``num_learning_epochs`` passes reuse the same
partition. The STAR generator reshuffles (and re-resamples) every epoch, which is
both the more standard PPO behaviour and necessary for stochastic resampling to add
anything.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from gmtrack.pace import pace_env_split

STAR_GROUP = "star"
"""Observation group carrying ``[difficulty_weight, bin_id]`` per transition."""

TRACKING_FAILURES_EXTRA = "tracking_failures"
"""Environment-extra key carrying the true (non-timeout) tracking failure mask."""

ValidSampleMode = Literal["failure_prefix", "combined_done_prefix"]
"""Supported interpretations of PACE's effective-sample prefix."""


class StarBatch(RolloutStorage.Batch):
  """Mini-batch that knows which of its rows are acquisition transitions."""

  def __init__(self, *args, acq_mask: torch.Tensor | None = None, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.acq_mask: torch.Tensor | None = acq_mask
    """``(M,)`` bool. True for rows drawn from acquisition environments."""


class StarRolloutStorage(RolloutStorage):
  """Rollout storage that supports STAR resampling and PACE role masks.

  Args:
    acquisition_fraction: ``xi``. ``None`` (Stage I) means every environment is an
      acquisition environment and no consolidation batch is produced.
    rho_topk: Fraction of fragments retained per difficulty bin (paper: 0.05).
    rho_star: Fraction of each acquisition mini-batch drawn from the STAR pool
      (paper: 0.25).
    use_star: Set ``False`` for the "w/o STAR" ablation, or for Stage I.
  """

  def __init__(
    self,
    training_type: str,
    num_envs: int,
    num_transitions_per_env: int,
    obs: TensorDict,
    actions_shape: tuple[int, ...] | list[int],
    device: str = "cpu",
    acquisition_fraction: float | None = None,
    rho_topk: float = 0.05,
    rho_star: float = 0.25,
    use_star: bool = True,
  ) -> None:
    super().__init__(
      training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
    )
    self.acquisition_fraction = acquisition_fraction
    self.rho_topk = rho_topk
    self.rho_star = rho_star
    self.use_star = use_star

    self.env_split = (
      num_envs
      if acquisition_fraction is None
      else pace_env_split(acquisition_fraction, num_envs)
    )

    flat = torch.arange(num_transitions_per_env * num_envs, device=device)
    env_of_flat = flat % num_envs
    self.acq_flat_idx = flat[env_of_flat < self.env_split]
    self.con_flat_idx = flat[env_of_flat >= self.env_split]

    if training_type == "rl":
      # Advantages before the H/E normalization; fragment scoring needs raw values.
      self.raw_advantages = torch.zeros(
        num_transitions_per_env, num_envs, 1, device=device
      )

    # ``dones`` is the Gymnasium union of terminated and truncated. PACE progress is
    # about tracking failures, so keep that signal separately: clip ends and episode
    # time limits must still terminate GAE/STAR fragments without shortening Eq. 17's
    # effective-sample prefix.
    self.tracking_failures = torch.zeros(
      num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device
    )
    self.tracking_failure_sources = torch.zeros(
      num_transitions_per_env, dtype=torch.uint8, device=device
    )
    """Per-step source: 0 missing, 1 explicit env mask, 2 derived compatibility mask."""

    self.last_star_pool_size = 0
    """Diagnostic: |P| from the most recent generator call."""

  # -- STAR bookkeeping -----------------------------------------------------

  def _star_meta(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened ``(difficulty weight, bin id)`` per transition."""
    meta = self.observations[STAR_GROUP].flatten(0, 1)
    return meta[:, 0], meta[:, 1].long()

  # -- PACE valid-sample bookkeeping ---------------------------------------

  def record_tracking_failures(
    self, failures: torch.Tensor, *, derived: bool = False
  ) -> None:
    """Record the true tracking-failure mask for the transition being appended.

    Call this immediately before :meth:`add_transition`. ``derived=True`` identifies
    the compatibility path ``done & ~time_out``; an explicit terminated mask is
    preferable because terminated and truncated may both be true on the same step.
    """
    if self.step >= self.num_transitions_per_env:
      raise OverflowError(
        "Rollout buffer overflow while recording tracking failures. Call clear() "
        "before adding new transitions."
      )
    mask = failures.to(device=self.device, dtype=torch.bool).reshape(-1)
    if mask.shape != (self.num_envs,):
      raise ValueError(
        "tracking failure mask must contain one value per environment; "
        f"expected {(self.num_envs,)}, got {tuple(mask.shape)}."
      )
    self.tracking_failures[self.step, :, 0].copy_(mask)
    self.tracking_failure_sources[self.step] = 2 if derived else 1

  def clear(self) -> None:
    """Reset the rollout cursor and failure-mask coverage diagnostics."""
    super().clear()
    self.tracking_failures.zero_()
    self.tracking_failure_sources.zero_()

  @staticmethod
  def _prefix_before_first_event(events: torch.Tensor) -> torch.Tensor:
    """Include every row up to and including the first event in each environment."""
    prior_events = torch.zeros_like(events, dtype=torch.long)
    prior_events[1:] = torch.cumsum(events.long(), dim=0)[:-1]
    return prior_events == 0

  def valid_sample_diagnostics(
    self, mode: ValidSampleMode = "failure_prefix"
  ) -> dict[str, int | str]:
    """Return PACE counts plus failure/done provenance for audit logging.

    ``failure_prefix`` is the paper implementation used by default. The legacy
    ``combined_done_prefix`` mode remains available to reproduce older checkpoints'
    logging and to quantify how much timeout/motion-end truncation biased ``rho``.
    Neither mode changes which rows enter PPO, value, entropy, or Eq. 15 losses.
    """
    if mode not in ("failure_prefix", "combined_done_prefix"):
      raise ValueError(
        "valid sample mode must be 'failure_prefix' or 'combined_done_prefix', "
        f"got {mode!r}."
      )

    populated_steps = self.num_transitions_per_env if self.step == 0 else self.step
    sources = self.tracking_failure_sources[:populated_steps]
    if mode == "failure_prefix" and bool((sources == 0).any()):
      missing = int((sources == 0).sum().item())
      raise RuntimeError(
        f"Missing tracking failure masks for {missing}/{populated_steps} rollout "
        f"steps. Pass extras[{TRACKING_FAILURES_EXTRA!r}] from the environment's "
        "terminated signal (preferred), or provide time_outs for compatibility "
        "derivation."
      )

    failures = self.tracking_failures[:populated_steps].squeeze(-1)
    dones = self.dones[:populated_steps].squeeze(-1).bool()
    valid_failure = self._prefix_before_first_event(failures)
    valid_combined = self._prefix_before_first_event(dones)
    valid = valid_failure if mode == "failure_prefix" else valid_combined
    non_failure_dones = dones & ~failures

    def role_counts(mask: torch.Tensor) -> tuple[int, int]:
      return (
        int(mask[:, : self.env_split].sum().item()),
        int(mask[:, self.env_split :].sum().item()),
      )

    n_acq, n_con = role_counts(valid)
    failure_acq, failure_con = role_counts(failures)
    done_acq, done_con = role_counts(dones)
    non_failure_done_acq, non_failure_done_con = role_counts(non_failure_dones)
    legacy_acq, legacy_con = role_counts(valid_combined)
    failure_prefix_acq, failure_prefix_con = role_counts(valid_failure)
    return {
      "mode": mode,
      "valid_acq_samples": n_acq,
      "valid_con_samples": n_con,
      "failure_prefix_valid_acq_samples": failure_prefix_acq,
      "failure_prefix_valid_con_samples": failure_prefix_con,
      "combined_done_prefix_valid_acq_samples": legacy_acq,
      "combined_done_prefix_valid_con_samples": legacy_con,
      "valid_acq_gain_vs_combined_done": failure_prefix_acq - legacy_acq,
      "valid_con_gain_vs_combined_done": failure_prefix_con - legacy_con,
      "tracking_failure_acq_events": failure_acq,
      "tracking_failure_con_events": failure_con,
      "combined_done_acq_events": done_acq,
      "combined_done_con_events": done_con,
      "non_failure_done_acq_events": non_failure_done_acq,
      "non_failure_done_con_events": non_failure_done_con,
      "tracking_failure_explicit_steps": int((sources == 1).sum().item()),
      "tracking_failure_derived_steps": int((sources == 2).sum().item()),
      "tracking_failure_missing_steps": int((sources == 0).sum().item()),
    }

  def fragment_ids(self) -> torch.Tensor:
    """Unique id of the maximal contiguous run each transition belongs to.

    A fragment ends at a ``done``, so the fragment counter within an environment is
    the exclusive cumulative sum of the done flags.
    """
    dones = self.dones.squeeze(-1).float()  # (T, N)
    counter = torch.zeros_like(dones)
    counter[1:] = torch.cumsum(dones, dim=0)[:-1]
    env_ids = torch.arange(self.num_envs, device=self.device).expand_as(counter)
    return (env_ids * (self.num_transitions_per_env + 1) + counter.long()).flatten()

  def _group_moments(
    self, values: torch.Tensor, mask: torch.Tensor, distributed: bool
  ) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Return global ``(count, mean, unbiased std)`` for a masked group."""
    selected = values[mask]
    stats = torch.stack(
      (
        torch.tensor(float(selected.numel()), device=values.device),
        selected.sum(),
        selected.square().sum(),
      )
    )
    if distributed:
      if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
          "Distributed advantage normalization requested before torch.distributed "
          "was initialized."
        )
      torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    count = int(stats[0].item())
    if count == 0:
      zero = values.new_zeros(())
      return 0, zero, zero
    mean = stats[1] / count
    if count < 2:
      return count, mean, values.new_zeros(())
    variance = (stats[2] - count * mean.square()).clamp_min(0.0) / (count - 1)
    return count, mean, variance.sqrt()

  def normalize_acquisition_advantages(
    self, eps: float = 1e-8, distributed: bool = False
  ) -> None:
    """Normalize over acquisition rows only and zero unused consolidation rows."""
    adv = self.advantages.flatten(0, 1).squeeze(-1)
    acq = torch.zeros_like(adv, dtype=torch.bool)
    acq[self.acq_flat_idx] = True
    count, mean, std = self._group_moments(adv, acq, distributed)
    if count < 2:
      raise RuntimeError("Advantage normalization needs at least two acquisition rows.")
    normalized = torch.zeros_like(adv)
    normalized[acq] = (adv[acq] - mean) / (std + eps)
    self.advantages.copy_(
      normalized.view(self.num_transitions_per_env, self.num_envs, 1)
    )

  def normalize_advantages_by_difficulty(
    self, eps: float = 1e-8, distributed: bool = False
  ) -> None:
    """Eq. (23): normalize inside ``H = {w > 1}`` and ``E = {w <= 1}`` independently.

    Early in training no bin has accumulated failures yet, so ``H`` can be empty or
    near-empty. Splitting then is meaningless (and a 1-sample std is undefined), so
    this falls back to a single acquisition-wide normalization -- one branch or the
    other, never both, so acquisition advantages are normalized exactly once.
    """
    adv = self.advantages.flatten(0, 1).squeeze(-1)
    weights, _ = self._star_meta()
    acq = torch.zeros_like(adv, dtype=torch.bool)
    acq[self.acq_flat_idx] = True
    high = acq & (weights > 1.0)
    low = acq & ~high

    high_stats = self._group_moments(adv, high, distributed)
    low_stats = self._group_moments(adv, low, distributed)
    normalized = torch.zeros_like(adv)
    if high_stats[0] < 2 or low_stats[0] < 2:
      count, mean, std = self._group_moments(adv, acq, distributed)
      if count < 2:
        raise RuntimeError(
          "Advantage normalization needs at least two acquisition rows."
        )
      normalized[acq] = (adv[acq] - mean) / (std + eps)
    else:
      for mask, (_, mean, std) in ((high, high_stats), (low, low_stats)):
        normalized[mask] = (adv[mask] - mean) / (std + eps)

    self.advantages.copy_(
      normalized.view(self.num_transitions_per_env, self.num_envs, 1)
    )

  # -- STAR pool ------------------------------------------------------------

  def _build_star_pool(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pool_indices, sampling_weights)``; both empty if STAR cannot apply."""
    empty = torch.empty(0, dtype=torch.long, device=self.device)
    if not self.use_star:
      return empty, empty.float()

    weights, bin_ids = self._star_meta()
    raw_adv = self.raw_advantages.flatten(0, 1).squeeze(-1)
    frags = self.fragment_ids()

    acq = torch.zeros_like(weights, dtype=torch.bool)
    acq[self.acq_flat_idx] = True
    high = acq & (weights > 1.0)
    if int(high.sum()) == 0:
      return empty, empty.float()

    # Group high-difficulty transitions by (bin, fragment) -> q_{b,tau} (Eq. 25).
    stride = int(frags.max().item()) + 1
    keys = bin_ids[high] * stride + frags[high]
    uniq_keys, inverse = torch.unique(keys, return_inverse=True)
    counts = torch.zeros(uniq_keys.numel(), device=self.device)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float))
    sums = torch.zeros(uniq_keys.numel(), device=self.device)
    sums.index_add_(0, inverse, raw_adv[high])
    q = sums / counts.clamp(min=1.0)

    uniq_bin = torch.div(uniq_keys, stride, rounding_mode="floor")
    uniq_frag = uniq_keys % stride

    # Rank pairs by q descending *within* each bin (Eq. 26), keep the top k_b.
    by_q = torch.argsort(q, descending=True)
    order = by_q[torch.argsort(uniq_bin[by_q], stable=True)]
    bins_sorted = uniq_bin[order]
    _, seg_counts = torch.unique_consecutive(bins_sorted, return_counts=True)
    seg_starts = torch.cumsum(seg_counts, 0) - seg_counts
    rank = torch.arange(order.numel(), device=self.device) - torch.repeat_interleave(
      seg_starts, seg_counts
    )
    k_b = torch.clamp(torch.ceil(self.rho_topk * seg_counts.float()).long(), min=1)
    keep = rank < torch.repeat_interleave(k_b, seg_counts)

    selected_frags = torch.unique(uniq_frag[order][keep])
    if selected_frags.numel() == 0:
      return empty, empty.float()

    # P = every acquisition transition inside a selected fragment (Eq. 27).
    in_pool = acq & torch.isin(frags, selected_frags)
    pool = torch.nonzero(in_pool, as_tuple=False).squeeze(-1)
    if pool.numel() == 0:
      return empty, empty.float()

    # eta_tau = mean difficulty weight over the fragment (Eq. 29).
    pool_frags = frags[pool]
    uniq_pf, pf_inv = torch.unique(pool_frags, return_inverse=True)
    pf_counts = torch.zeros(uniq_pf.numel(), device=self.device)
    pf_counts.index_add_(0, pf_inv, torch.ones_like(pf_inv, dtype=torch.float))
    pf_sums = torch.zeros(uniq_pf.numel(), device=self.device)
    pf_sums.index_add_(0, pf_inv, weights[pool])
    eta = pf_sums / pf_counts.clamp(min=1.0)

    omega = eta[pf_inv]  # (|P|,) -- Eq. (30) normalizes; multinomial does that for us.
    if float(omega.sum()) <= 0:
      return empty, empty.float()
    return pool, omega

  # -- generators -----------------------------------------------------------

  def star_mini_batch_generator(
    self, num_mini_batches: int, num_epochs: int = 8
  ) -> Generator[StarBatch, None, None]:
    """Mixed acquisition/consolidation mini-batches with STAR resampling.

    Each yielded batch is ``[acquisition rows ; consolidation rows]`` with
    ``acq_mask`` marking the split. Row counts follow ``xi`` so the compute per update
    matches the vanilla generator.
    """
    if self.training_type != "rl":
      raise ValueError("Only available for reinforcement-learning training.")

    total = self.num_envs * self.num_transitions_per_env
    mini_batch_size = total // num_mini_batches
    n_acq_pool = int(self.acq_flat_idx.numel())
    n_con_pool = int(self.con_flat_idx.numel())

    if n_con_pool == 0:
      acq_per_batch, con_per_batch = mini_batch_size, 0
    else:
      if mini_batch_size < 2:
        raise ValueError(
          f"mini_batch_size={mini_batch_size} cannot hold both an acquisition and a "
          f"consolidation row. Reduce num_mini_batches or raise num_envs."
        )
      # Clamped to leave room for at least one consolidation row. Without the upper
      # clamp, a lopsided split rounds acq up to the full batch and the consolidation
      # row is then *added* on top, silently making the batch one row larger than
      # every other update's.
      acq_per_batch = min(
        max(int(round(mini_batch_size * n_acq_pool / total)), 1), mini_batch_size - 1
      )
      con_per_batch = mini_batch_size - acq_per_batch
    assert acq_per_batch + con_per_batch == mini_batch_size or n_con_pool == 0

    pool, omega = self._build_star_pool()
    self.last_star_pool_size = int(pool.numel())
    m_star = (
      min(acq_per_batch, max(int(self.rho_star * acq_per_batch), 1))
      if pool.numel() > 0
      else 0
    )

    observations = self.observations.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_params = tuple(p.flatten(0, 1) for p in self.distribution_params)

    acq_mask_template = torch.zeros(
      acq_per_batch + con_per_batch, dtype=torch.bool, device=self.device
    )
    acq_mask_template[:acq_per_batch] = True

    for _ in range(num_epochs):
      # Reshuffled every epoch (upstream shuffles once for all epochs).
      acq_perm = self.acq_flat_idx[torch.randperm(n_acq_pool, device=self.device)]
      con_perm = (
        self.con_flat_idx[torch.randperm(n_con_pool, device=self.device)]
        if n_con_pool
        else None
      )

      for i in range(num_mini_batches):
        acq_idx = _wrapped_slice(acq_perm, i * acq_per_batch, acq_per_batch)
        if m_star > 0:
          resampled = pool[torch.multinomial(omega, m_star, replacement=True)]
          acq_idx = torch.cat([resampled, acq_idx[m_star:]])

        if con_perm is not None:
          con_idx = _wrapped_slice(con_perm, i * con_per_batch, con_per_batch)
          batch_idx = torch.cat([acq_idx, con_idx])
        else:
          batch_idx = acq_idx

        yield StarBatch(
          observations=observations[batch_idx],
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_log_prob[batch_idx],
          old_distribution_params=tuple(p[batch_idx] for p in old_params),
          acq_mask=acq_mask_template,
        )

  def valid_sample_counts(
    self, mode: ValidSampleMode = "failure_prefix"
  ) -> tuple[int, int]:
    """``(N_A, N_C)`` for PACE's progress ratio (Eq. 17).

    The default includes steps at or before the first *tracking failure*. Timeouts and
    normal motion ends remain valid. Pass ``"combined_done_prefix"`` only for legacy
    reproduction/diagnosis; it implements the old prefix ending on any ``done``.
    """
    diagnostics = self.valid_sample_diagnostics(mode)
    return (
      int(diagnostics["valid_acq_samples"]),
      int(diagnostics["valid_con_samples"]),
    )


def _wrapped_slice(perm: torch.Tensor, start: int, length: int) -> torch.Tensor:
  """Take ``length`` entries from ``perm`` starting at ``start``, wrapping around.

  The acquisition and consolidation pools have different sizes, so a plain slice can
  run past the end of the shorter one on the last mini-batch.
  """
  n = perm.numel()
  if length <= 0:
    return perm[:0]
  if n == 0:
    raise ValueError("Cannot sample from an empty index pool.")
  idx = torch.arange(start, start + length, device=perm.device) % n
  return perm[idx]
