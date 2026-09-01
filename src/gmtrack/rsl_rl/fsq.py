"""Finite Scalar Quantization bottleneck (Mentzer et al., 2023).

Paper Sec. IV-A / Eq. (10): the history-conditioned command feature ``u_t`` is
factorized into two 32-dimensional tokens and quantized before it reaches the actor.
Applying the bottleneck *after* command aggregation (rather than to the raw reference
inputs) is what limits the policy's sensitivity to local inconsistencies in the
reference trajectory -- exactly the failure mode of Xsens inertial capture, which is
noisy in root drift, local pose and contact timing.

ASSUMPTION: the paper states the token layout (2 x 32) but not the number of
quantization levels per dimension. Following the SONIC training release under
``cankao/GR00T-WholeBodyControl``, whose universal tokenizer uses the same 2 x 32
layout, we use 32 levels per dimension as the unpublished-detail proxy. It remains a
config knob (``fsq_levels``) and is not claimed as a GMTrack paper value.
"""

from __future__ import annotations

import torch
from torch import nn

SONIC_PROXY_FSQ_LEVELS = 32
"""Per-dimension levels used by SONIC's published 2 x 32 universal tokenizer."""


def round_ste(x: torch.Tensor) -> torch.Tensor:
  """Round with a straight-through gradient estimator."""
  return x + (torch.round(x) - x).detach()


class FSQ(nn.Module):
  """Element-wise finite scalar quantization with a straight-through estimator.

  Args:
    dim: Total feature width. Must be divisible by ``token_dim``.
    levels: Number of quantization levels per dimension.
    token_dim: Width of one token. The paper uses 32, giving 2 tokens over a
      64-d feature. Only affects bookkeeping/diagnostics -- quantization itself is
      element-wise, so the grouping does not change the forward pass.
    eps: Expands the pre-rounding range slightly so both extreme integer codes stay
      reachable, matching SONIC's ``vector-quantize-pytorch`` implementation.

  Shape:
    - input: ``(..., dim)``
    - output: ``(..., dim)``, values on a discrete grid in ``[-1, 1]``.
  """

  def __init__(
    self,
    dim: int,
    levels: int = SONIC_PROXY_FSQ_LEVELS,
    token_dim: int = 32,
    eps: float = 1e-3,
  ) -> None:
    super().__init__()
    if levels < 2:
      raise ValueError(f"FSQ needs at least 2 levels, got {levels}.")
    if token_dim > 0 and dim % token_dim != 0:
      raise ValueError(f"dim={dim} is not divisible by token_dim={token_dim}.")

    self.dim = dim
    self.levels = levels
    self.token_dim = token_dim
    self.num_tokens = dim // token_dim if token_dim > 0 else 1

    # Match the vector-quantize-pytorch FSQ used by SONIC's training release.
    half_l = (levels - 1) * (1.0 + eps) / 2.0
    # Even level counts use FSQ's half-step offset; the inverse-tanh shift keeps a
    # zero input exactly on the zero code before rounding.
    offset = 0.5 if levels % 2 == 0 else 0.0
    self.register_buffer("_half_l", torch.tensor(half_l), persistent=False)
    self.register_buffer("_offset", torch.tensor(offset), persistent=False)
    self.register_buffer(
      "_shift", torch.atanh(torch.tensor(offset / half_l)), persistent=False
    )
    self.register_buffer(
      "_half_width", torch.tensor(float(levels // 2)), persistent=False
    )

  def bound(self, z: torch.Tensor) -> torch.Tensor:
    """Squash to the open interval covering all quantization levels."""
    return torch.tanh(z + self._shift) * self._half_l - self._offset

  def forward(self, z: torch.Tensor) -> torch.Tensor:
    quantized = round_ste(self.bound(z))
    return quantized / self._half_width

  @torch.no_grad()
  def code_indices(self, z: torch.Tensor) -> torch.Tensor:
    """Per-dimension integer codes in ``[0, levels)``, for diagnostics.

    A global codebook index is deliberately not returned: with 32 dimensions per
    token, the implicit product codebook far exceeds int64 for the SONIC proxy.
    """
    quantized = torch.round(self.bound(z))
    return (quantized + self._half_width).long().clamp_(0, self.levels - 1)

  @torch.no_grad()
  def usage_entropy(self, z: torch.Tensor) -> torch.Tensor:
    """Mean normalized per-dimension code entropy over a batch.

    Near 1 means every level is being used; a collapse toward 0 means the bottleneck
    has degenerated into a constant and the command signal is no longer reaching the
    actor.
    """
    codes = self.code_indices(z).reshape(-1, self.dim)
    onehot = torch.zeros(
      codes.shape[0], self.dim, self.levels, device=z.device, dtype=z.dtype
    )
    onehot.scatter_(2, codes.unsqueeze(-1), 1.0)
    p = onehot.mean(dim=0)  # (dim, levels)
    h = -(p * (p + 1e-12).log()).sum(dim=-1)
    return (h / torch.log(torch.tensor(float(self.levels), device=z.device))).mean()

  def extra_repr(self) -> str:
    return (
      f"dim={self.dim}, levels={self.levels}, tokens={self.num_tokens}x{self.token_dim}"
    )
