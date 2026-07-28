"""Adaptive difficulty-bin sampling over a set of motion clips.

Paper Eq. (12)-(13): every motion sequence is partitioned into temporal bins; each
bin keeps an exponential moving average of failure events, and reset states are drawn
from a normalised, uniform-mixed distribution over those bins so that hard segments
get visited more often.

mjlab implements this for a *single* clip inside ``MotionCommand._adaptive_sampling``.
Extreme-RGMT applies it to a whole motion set (the full set ``D`` in Stage I, the
challenging set ``D_c`` in Stage II's acquisition environments), so the bin
distribution here spans every (clip, bin) pair in the subset.

The same object also serves STAR (Sec. V-B): ``bin_weight`` returns
``w_t = B * p_{b_t}``, the difficulty weight of a transition relative to the uniform
baseline ``1/B``.
"""

from __future__ import annotations

import math

import torch


class AdaptiveBinSampler:
  """Failure-driven categorical distribution over (clip, temporal bin) pairs.

  Bins are laid out as a right-padded ``(num_clips, max_bins)`` grid so the
  smoothing kernel can be applied per clip without bleeding across clip
  boundaries -- a flat 1-D layout would smear the tail of one clip into the head
  of the next.

  Args:
    clip_ids: ``(C,)`` library clip ids this sampler draws from.
    clip_bins: ``(C,)`` number of valid bins per clip.
    max_bins: Width of the padded bin axis.
    kernel_size: Non-causal smoothing window, in bins. 1 disables smoothing.
    kernel_lambda: Geometric decay of the smoothing kernel.
    uniform_ratio: ``eps_u`` in Eq. (13); mass reserved for the uniform baseline.
    alpha: EMA rate in Eq. (12).
    max_count: ``c_max`` in Eq. (13); failure counts are clipped here before
      normalisation so a single pathological bin cannot capture all sampling mass.
    device: Torch device.
  """

  def __init__(
    self,
    clip_ids: torch.Tensor,
    clip_bins: torch.Tensor,
    max_bins: int,
    num_library_clips: int,
    kernel_size: int = 1,
    kernel_lambda: float = 0.8,
    uniform_ratio: float = 0.1,
    alpha: float = 0.001,
    max_count: float = 1.0,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.clip_ids = clip_ids.to(device)
    self.num_clips = int(clip_ids.numel())
    self.max_bins = int(max_bins)
    if uniform_ratio <= 0.0:
      raise ValueError(
        "uniform_ratio must be > 0; it is the only term keeping the bin distribution "
        "positive before any failures have been recorded (Eq. 13)."
      )
    self.uniform_ratio = uniform_ratio
    self.alpha = alpha
    self.max_count = max_count

    bins = clip_bins.to(device)
    self.valid = torch.arange(self.max_bins, device=device)[None, :] < bins[:, None]
    self.num_valid_bins = int(self.valid.sum().item())

    self.failed_ema = torch.zeros(self.num_clips, self.max_bins, device=device)
    self._pending = torch.zeros(self.num_clips, self.max_bins, device=device)

    self.kernel_size = max(int(kernel_size), 1)
    kernel = torch.tensor(
      [kernel_lambda**i for i in range(self.kernel_size)], device=device
    )
    self.kernel = (kernel / kernel.sum()).view(1, 1, -1)

    # Reverse lookup: library clip id -> row in this sampler (-1 when absent).
    # Sized to the *whole* library, not this subset: callers legitimately query with
    # clip ids belonging to the other PACE group (see `bin_weight`), and sizing this
    # by `clip_ids.max()` would turn that into an out-of-bounds read.
    self.row_of_clip = torch.full(
      (num_library_clips,), -1, dtype=torch.long, device=device
    )
    self.row_of_clip[self.clip_ids] = torch.arange(
      self.num_clips, dtype=torch.long, device=device
    )

    self._probs = self._compute_probs()

  # -- distribution ---------------------------------------------------------

  def _compute_probs(self) -> torch.Tensor:
    """Normalised ``(C, max_bins)`` sampling distribution, zero on padding.

    Eq. (13) in order: ``s_i = Normalize(clip(c_i, 0, c_max))`` **first**, then
    ``p_i ∝ s_i + eps_u / N``.

    The order matters. Normalizing the clipped counts to sum 1 before mixing fixes
    the uniform baseline's share at ``eps_u / (1 + eps_u)`` for the whole run. Adding
    ``eps_u / N`` to the *unnormalized* counts instead (the mjlab formulation) makes
    that share depend on how many failures have accumulated: near-uniform early on,
    then progressively more peaked as the EMA grows -- an unintended implicit
    schedule on the exploration/exploitation balance.
    """
    scores = torch.clamp(self.failed_ema, 0.0, self.max_count) * self.valid

    total = scores.sum()
    if total > 0:
      scores = scores / total
    # else: no failures recorded yet, so s_i = 0 and the uniform baseline alone
    # defines the distribution -- which is the correct behaviour at initialization.

    scores = scores + self.uniform_ratio / max(self.num_valid_bins, 1)
    scores = scores * self.valid

    if self.kernel_size > 1:
      # NOT in the paper: inherited from mjlab, which convolves a small non-causal
      # kernel over the bin axis to spread credit into neighbouring bins. Disabled by
      # default (`adaptive_kernel_size=1`); kept because it is occasionally useful
      # when bins are narrow relative to the failure horizon.
      padded = torch.nn.functional.pad(
        scores.unsqueeze(1), (0, self.kernel_size - 1), mode="replicate"
      )
      scores = torch.nn.functional.conv1d(padded, self.kernel).squeeze(1)
      scores = scores * self.valid

    # `uniform_ratio > 0` guarantees a positive total over the valid bins, so a
    # non-positive sum means the config is broken and should surface as an error.
    return scores / scores.sum()

  @property
  def probs(self) -> torch.Tensor:
    return self._probs

  def refresh(self) -> None:
    """Recompute the distribution after the EMA moved."""
    self._probs = self._compute_probs()

  def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n`` (library clip id, bin index) pairs with replacement."""
    flat = torch.multinomial(self._probs.reshape(-1), n, replacement=True)
    rows = torch.div(flat, self.max_bins, rounding_mode="floor")
    bins = flat % self.max_bins
    return self.clip_ids[rows], bins

  def sample_uniform(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``n`` pairs uniformly over valid bins (consolidation environments)."""
    flat = torch.multinomial(self.valid.reshape(-1).float(), n, replacement=True)
    rows = torch.div(flat, self.max_bins, rounding_mode="floor")
    bins = flat % self.max_bins
    return self.clip_ids[rows], bins

  # -- failure bookkeeping --------------------------------------------------

  def record_failures(self, clip_ids: torch.Tensor, bins: torch.Tensor) -> None:
    """Accumulate this step's failures, to be folded in by :meth:`step_ema`."""
    if clip_ids.numel() == 0:
      return
    rows = self.row_of_clip[clip_ids]
    keep = rows >= 0
    if not bool(keep.any()):
      return
    flat = rows[keep] * self.max_bins + bins[keep]
    self._pending.view(-1).index_add_(
      0, flat, torch.ones_like(flat, dtype=self._pending.dtype)
    )

  def step_ema(self) -> None:
    """Apply Eq. (12) and clear the per-step accumulator."""
    self.failed_ema.mul_(1.0 - self.alpha).add_(self._pending, alpha=self.alpha)
    self._pending.zero_()
    self.refresh()

  # -- STAR hooks -----------------------------------------------------------

  def bin_weight(self, clip_ids: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """``w_t = B * p_{b_t}`` (Eq. 20). ``>1`` means above the uniform baseline.

    Environments whose clip is not part of this sampler's subset get weight 0, which
    puts them in STAR's low-difficulty group ``E`` and excludes them from fragment
    selection.
    """
    rows = self.row_of_clip[clip_ids]
    inside = rows >= 0
    safe_rows = torch.where(inside, rows, torch.zeros_like(rows))
    p = self._probs[safe_rows, bins]
    return torch.where(inside, p * self.num_valid_bins, torch.zeros_like(p))

  def flat_bin_id(self, clip_ids: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Unique id per (clip, bin) pair, for STAR's per-bin fragment ranking.

    Clips outside this subset map to row 0. That is harmless because STAR only groups
    transitions in ``H = {w > 1}``, and :meth:`bin_weight` gives those clips weight 0.
    """
    rows = self.row_of_clip[clip_ids]
    rows = torch.where(rows >= 0, rows, torch.zeros_like(rows))
    return rows * self.max_bins + bins

  # -- logging --------------------------------------------------------------

  def entropy_stats(self) -> tuple[float, float]:
    """Normalised entropy and top-1 probability of the current distribution."""
    p = self._probs.reshape(-1)
    p = p[p > 0]
    h = float(-(p * p.log()).sum().item())
    h_norm = h / math.log(self.num_valid_bins) if self.num_valid_bins > 1 else 1.0
    return h_norm, float(self._probs.max().item())
