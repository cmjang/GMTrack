"""Extreme-RGMT policy architecture (paper Sec. IV-A, Fig. 3, Table III).

Data flow::

  proprio history  (N, H*64)  -> f_o -> LN -> z^o  (N, H, 64)     Eq. (5)
  action  history  (N, H*29)  -> f_a -> LN -> z^a  (N, H, 64)     Eq. (5)
  interleave [z^a_{t-H-1}, z^o_{t-H}, ..., z^a_{t-1}, z^o_t]      Eq. (6)
     + positional encoding -> pre-LN causal block -> max-pool
     -> h_t (N, 64)                       Eq. (7) / RGMT Eq. (10-11)

  command window   (N, 21*C) -> f_g -> LN + p_tau -> Z^g (N, 21, 64)  Eq. (8)

  u_t   = pre-LN cross block(Q = W_q h_t, K/V = Z^g)  (N, 64)
                                          Eq. (9) / RGMT Eq. (15)
  u_hat = FSQ(u_t)                                                 Eq. (10)
  a_t   = pi(o^prop_t, a_{t-1}, u_hat)                             Eq. (11)

Extreme-RGMT leaves both attention blocks' internals unspecified; they come from
RGMT (arXiv:2601.23080v1), whose Eq. 10-11 and Eq. 15 spell out pre-LN residual
sublayers (MHA + MLP), a final LayerNorm, and element-wise max pooling over time
for the history branch. These details are load-bearing: with a plain post-LN
attention layer instead, the actor's output moves so much per optimizer step that
the adaptive-KL schedule floors the learning rate permanently (verified by
training probes; the paper authors confirmed they train with the adaptive schedule
exactly as published).

Note that Eq. (8) defines ``Z^g`` as *encoded* command tokens only -- the command
branch has no self-attention of its own, it supplies keys and values directly. We
follow the equations rather than the block diagram, which is ambiguous on this point.

The class plugs into rsl-rl through ``class_name``; ``resolve_callable`` accepts a
``"module:Class"`` string, so no rsl-rl source change is needed. Extra constructor
kwargs flow in from ``RslRlModelCfg`` via ``**cfg["actor"]``.
"""

from __future__ import annotations

import copy
import math

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict
from torch import nn

from ex_grmt.rsl_rl.fsq import FSQ, SONIC_PROXY_FSQ_LEVELS

# Observation groups this model expects, in this order. The env cfg
# (``ex_grmt.envs.stage1_env_cfg``) is responsible for producing them.
PROPRIO_HIST = "proprio_hist"
ACTION_HIST = "action_hist"
COMMAND_WINDOW = "command_window"
REQUIRED_GROUPS = (PROPRIO_HIST, ACTION_HIST, COMMAND_WINDOW)


class ExGRMTActor(MLPModel):
  """Dynamics-guided command-encoding actor.

  Args:
    obs / obs_groups / obs_set / output_dim: rsl-rl model contract. ``obs_groups[obs_set]``
      must be exactly :data:`REQUIRED_GROUPS`.
    hidden_dims: Actor trunk. Paper Table III: ``(1024, 1024, 512, 256)``.
    activation: Trunk activation.
    obs_normalization: Leave ``False``. The paper explicitly replaces running-statistic
      observation normalization with per-branch LayerNorm, because empirical statistics
      drift badly when the motion distribution shifts between Stage I and Stage II.
    distribution_cfg: Passed through to rsl-rl (``GaussianDistribution``).
    history_length: ``H``. Paper uses a 10-frame proprioceptive and action history.
    proprio_term_dims: Per-term widths inside the ``proprio_hist`` group, in the order
      the env cfg declares them. mjlab flattens history **per term**, so the group is
      laid out ``[g(H*3), w(H*3), q(H*29), qd(H*29)]`` -- not ``H`` interleaved frames.
      Getting this wrong silently scrambles the history.
    token_dim: Width of every branch embedding and of ``u_t``. Paper: 64.
    command_token_dim: Width ``C`` of one reference token. The original policy uses
      38 = ``v_ref(3) + w_ref(3) + g_ref(3) + q_ref(29)``. Heading closed-loop
      policies use 44 by appending ``root_ori_error(6)``.
    state_encoder_hidden / action_encoder_hidden / command_encoder_hidden: Hidden widths
      of ``f_o`` / ``f_a`` / ``f_g``. Table III lists ``[64,128,64]``, ``[29,64,64]``
      and ``[C,128,64]`` (input, hidden, output).
    num_heads: Attention heads. ASSUMPTION -- not stated in the paper.
    use_fsq: Set ``False`` for the "w/o FSQ" ablation (Table VIII).
    fsq_levels / fsq_token_dim: See :class:`~ex_grmt.rsl_rl.fsq.FSQ`.
    unified_encoder: Set ``True`` for the "Unified Enc." ablation (Table VIII): one
      shared encoder + normalization for proprioceptive and action histories instead
      of two separate branches.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (1024, 1024, 512, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    history_length: int = 10,
    proprio_term_dims: tuple[int, ...] = (3, 3, 29, 29),
    token_dim: int = 64,
    state_encoder_hidden: tuple[int, ...] = (128,),
    action_encoder_hidden: tuple[int, ...] = (64,),
    command_encoder_hidden: tuple[int, ...] = (128,),
    encoder_activation: str = "elu",
    num_heads: int = 4,
    use_fsq: bool = True,
    fsq_levels: int = SONIC_PROXY_FSQ_LEVELS,
    fsq_token_dim: int = 32,
    unified_encoder: bool = False,
    command_token_dim: int = 38,
  ) -> None:
    # Plain attributes are safe to set before nn.Module.__init__ runs; they are
    # needed by _get_obs_dim / _get_latent_dim, both called from super().__init__.
    self.history_length = history_length
    self.proprio_term_dims = tuple(proprio_term_dims)
    self.token_dim = token_dim
    self.command_token_dim = command_token_dim
    self.action_dim = output_dim
    self.use_fsq = use_fsq
    self.unified_encoder = unified_encoder

    if obs_normalization:
      raise ValueError(
        "ExGRMTActor uses per-branch LayerNorm instead of EmpiricalNormalization "
        "(paper Sec. IV-A); set obs_normalization=False."
      )
    if not isinstance(command_token_dim, int) or isinstance(command_token_dim, bool):
      raise TypeError(
        f"command_token_dim must be an integer, got {type(command_token_dim).__name__}."
      )
    if command_token_dim <= 0:
      raise ValueError(f"command_token_dim must be positive, got {command_token_dim}.")

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=False,
      distribution_cfg=distribution_cfg,
    )

    # -- branch encoders (Eq. 5, Eq. 8) --
    if unified_encoder:
      self.unified_enc = MLP(
        self.proprio_dim + self.action_dim,
        token_dim,
        list(state_encoder_hidden),
        encoder_activation,
      )
      self.unified_norm = nn.LayerNorm(token_dim)
    else:
      self.state_enc = MLP(
        self.proprio_dim, token_dim, list(state_encoder_hidden), encoder_activation
      )
      self.state_norm = nn.LayerNorm(token_dim)
      self.action_enc = MLP(
        self.action_dim, token_dim, list(action_encoder_hidden), encoder_activation
      )
      self.action_norm = nn.LayerNorm(token_dim)

    self.command_enc = MLP(
      self.command_token_dim,
      token_dim,
      list(command_encoder_hidden),
      encoder_activation,
    )
    self.command_norm = nn.LayerNorm(token_dim)

    # -- positional encodings --
    # RGMT Eq. 9 / Eq. 14 specify *sinusoidal* positional encodings for both token
    # streams, and Extreme-RGMT never overrides that (Eq. 8 just says "temporal
    # positional embedding"), so under the inheritance rule the fixed encoding
    # governs. Buffers, not parameters: they are deterministic functions of shape.
    hist_tokens = history_length if unified_encoder else 2 * history_length
    self.num_hist_tokens = hist_tokens
    self.register_buffer(
      "hist_pos",
      sinusoidal_positional_encoding(hist_tokens, token_dim),
      persistent=False,
    )
    self.register_buffer(
      "command_pos",
      sinusoidal_positional_encoding(self.num_command_tokens, token_dim),
      persistent=False,
    )

    # -- causal history encoder (Eq. 7; block internals per RGMT Eq. 10-11) --
    # Extreme-RGMT leaves Enc_hist unspecified and "builds on the controller design
    # in [5]"; RGMT (arXiv:2601.23080v1) Eq. 10 defines it as a *pre-LN* residual
    # block -- H1 = H0 + MHA(LN(H0)), H2 = H1 + MLP(LN(H1)), Hbar = LN(H2) -- with
    # element-wise max pooling over time (Eq. 11). Pre-LN keeps the block's output
    # sensitivity to weight updates bounded; the earlier post-LN simplification made
    # every Adam step move the policy so far that the adaptive-KL schedule pinned
    # the learning rate at its 1e-5 floor for entire runs.
    self.hist_attn = nn.MultiheadAttention(
      token_dim, num_heads, batch_first=True, dropout=0.0
    )
    self.hist_ln_attn = nn.LayerNorm(token_dim)
    self.hist_ln_mlp = nn.LayerNorm(token_dim)
    # ASSUMPTION: RGMT does not give the block-MLP width; 4x is the transformer
    # convention.
    self.hist_mlp = MLP(token_dim, token_dim, [4 * token_dim], encoder_activation)
    self.hist_ln_out = nn.LayerNorm(token_dim)
    self.register_buffer(
      "_causal_mask",
      torch.triu(torch.ones(hist_tokens, hist_tokens, dtype=torch.bool), diagonal=1),
      persistent=False,
    )

    # -- cross attention (Eq. 9 supplies Q = W_q h_t; internals per RGMT Eq. 15) --
    # s1 = q + MHA(LN(q), Z), s2 = s1 + MLP(LN(s1)), u = LN(s2).
    self.query_proj = nn.Linear(token_dim, token_dim)
    self.cross_attn = nn.MultiheadAttention(
      token_dim, num_heads, batch_first=True, dropout=0.0
    )
    self.cross_ln_q = nn.LayerNorm(token_dim)
    self.cross_ln_mlp = nn.LayerNorm(token_dim)
    self.cross_mlp = MLP(token_dim, token_dim, [4 * token_dim], encoder_activation)
    self.cross_ln_out = nn.LayerNorm(token_dim)

    # -- FSQ bottleneck (Eq. 10) --
    self.fsq = (
      FSQ(token_dim, levels=fsq_levels, token_dim=fsq_token_dim)
      if use_fsq
      else nn.Identity()
    )

    self._last_u: torch.Tensor | None = None

  # -- rsl-rl hooks ---------------------------------------------------------

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    """Validate the group contract and derive per-branch dimensions."""
    active = list(obs_groups[obs_set])
    if tuple(active) != REQUIRED_GROUPS:
      raise ValueError(
        f"ExGRMTActor expects obs_groups[{obs_set!r}] == {list(REQUIRED_GROUPS)}, "
        f"got {active}. Order matters: dimensions are derived positionally."
      )
    for name in active:
      if obs[name].dim() != 2:
        raise ValueError(
          f"Group '{name}' must be 2-D (batch, dim); got {tuple(obs[name].shape)}. "
          f"Set flatten_history_dim=True on the observation group."
        )

    proprio_flat = int(obs[PROPRIO_HIST].shape[-1])
    action_flat = int(obs[ACTION_HIST].shape[-1])
    command_flat = int(obs[COMMAND_WINDOW].shape[-1])

    expected_proprio = self.history_length * sum(self.proprio_term_dims)
    if proprio_flat != expected_proprio:
      raise ValueError(
        f"'{PROPRIO_HIST}' is {proprio_flat}-d but history_length="
        f"{self.history_length} x proprio_term_dims={self.proprio_term_dims} implies "
        f"{expected_proprio}. Check the env's observation group."
      )
    self.proprio_dim = sum(self.proprio_term_dims)

    if action_flat % self.history_length != 0:
      raise ValueError(
        f"'{ACTION_HIST}' is {action_flat}-d, not divisible by history_length="
        f"{self.history_length}."
      )
    action_dim = action_flat // self.history_length
    if action_dim != self.action_dim:
      raise ValueError(
        f"'{ACTION_HIST}' implies action dim {action_dim} but the model outputs "
        f"{self.action_dim}."
      )

    if command_flat % self.command_token_dim != 0:
      raise ValueError(
        f"'{COMMAND_WINDOW}' is {command_flat}-d, not divisible by token width "
        f"command_token_dim={self.command_token_dim}. Check that the actor and "
        "environment use the same command layout."
      )
    self.num_command_tokens = command_flat // self.command_token_dim
    if self.num_command_tokens == 0:
      raise ValueError(f"'{COMMAND_WINDOW}' must contain at least one command token.")

    return active, proprio_flat + action_flat + command_flat

  def _get_latent_dim(self) -> int:
    """Actor trunk input: ``[o^prop_t, a_{t-1}, u_hat_t]`` (Eq. 11)."""
    return self.proprio_dim + self.action_dim + self.token_dim

  def update_normalization(self, obs: TensorDict) -> None:
    """No-op: normalization is handled by per-branch LayerNorm."""
    return

  # -- forward --------------------------------------------------------------

  def _proprio_sequence(self, flat: torch.Tensor) -> torch.Tensor:
    """``(N, H*sum(dims))`` -> ``(N, H, sum(dims))``.

    mjlab flattens history *within* each term, so the incoming layout is a
    concatenation of per-term blocks, each of which is time-major. Reshaping the
    whole row at once would interleave terms and timesteps incorrectly.
    """
    n = flat.shape[0]
    h = self.history_length
    parts, offset = [], 0
    for d in self.proprio_term_dims:
      block = flat[:, offset : offset + h * d]
      parts.append(block.view(n, h, d))
      offset += h * d
    return torch.cat(parts, dim=-1)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state  # Non-recurrent.

    o_seq = self._proprio_sequence(obs[PROPRIO_HIST])  # (N, H, P)
    n, h, _ = o_seq.shape
    a_seq = obs[ACTION_HIST].view(n, h, self.action_dim)  # (N, H, A)
    g_seq = obs[COMMAND_WINDOW].view(n, self.num_command_tokens, self.command_token_dim)

    # -- history tokens (Eq. 5-6) --
    if self.unified_encoder:
      tokens = self.unified_norm(self.unified_enc(torch.cat([o_seq, a_seq], dim=-1)))
    else:
      z_o = self.state_norm(self.state_enc(o_seq))  # (N, H, D)
      z_a = self.action_norm(self.action_enc(a_seq))  # (N, H, D)
      # Interleave as [z^a_{t-H-1}, z^o_{t-H}, ..., z^a_{t-1}, z^o_t] so the final
      # token is the current proprioceptive state.
      tokens = torch.stack([z_a, z_o], dim=2).reshape(n, 2 * h, self.token_dim)

    tokens = tokens + self.hist_pos

    # -- causal history encoder (Eq. 7; pre-LN block per RGMT Eq. 10-11) --
    normed = self.hist_ln_attn(tokens)
    attn_out, _ = self.hist_attn(
      normed, normed, normed, attn_mask=self._causal_mask, need_weights=False
    )
    x = tokens + attn_out
    x = x + self.hist_mlp(self.hist_ln_mlp(x))
    h_bar = self.hist_ln_out(x)
    h_t = h_bar.max(dim=1).values  # RGMT Eq. 11: element-wise max pool over time.

    # -- command tokens (Eq. 8) --
    z_g = self.command_norm(self.command_enc(g_seq)) + self.command_pos

    # -- cross attention (Eq. 9 / RGMT Eq. 15) --
    query = self.query_proj(h_t).unsqueeze(1)  # (N, 1, D)
    attn_out, _ = self.cross_attn(self.cross_ln_q(query), z_g, z_g, need_weights=False)
    s = query + attn_out
    s = s + self.cross_mlp(self.cross_ln_mlp(s))
    u_t = self.cross_ln_out(s).squeeze(1)  # (N, D)
    # Detached: this is only read by `bottleneck_entropy` for logging. Keeping the
    # graph-attached tensor would pin the whole autograd graph alive between updates
    # and makes the module un-deepcopy-able, which breaks ONNX export.
    self._last_u = u_t.detach()

    # -- FSQ (Eq. 10) --
    u_hat = self.fsq(u_t)

    # -- actor input (Eq. 11) --
    return torch.cat([o_seq[:, -1], a_seq[:, -1], u_hat], dim=-1)

  @torch.no_grad()
  def bottleneck_entropy(self) -> torch.Tensor | None:
    """Normalized FSQ code entropy from the most recent forward pass.

    Worth logging: a collapse toward 0 means the command bottleneck has stopped
    carrying information and the actor is effectively blind to the reference.
    """
    if not self.use_fsq or self._last_u is None:
      return None
    return self.fsq.usage_entropy(self._last_u)  # type: ignore[union-attr]

  # -- export ---------------------------------------------------------------

  def as_jit(self) -> nn.Module:
    return _ExportExGRMTActor(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _ExportExGRMTActor(self)


class _ExportExGRMTActor(nn.Module):
  """Deterministic, export-friendly copy of :class:`ExGRMTActor`.

  ``MLPModel``'s exporters deep-copy only ``obs_normalizer`` and ``mlp``, which would
  silently drop every encoder and the attention stack. This wrapper carries the whole
  forward pass and exposes the three observation groups as named ONNX inputs, which is
  also the interface the on-robot runtime has to fill.
  """

  is_recurrent: bool = False

  def __init__(self, model: ExGRMTActor) -> None:
    super().__init__()
    # `Distribution.update()` caches a `torch.distributions.Normal` built from
    # graph-attached tensors, so after any forward pass the distribution submodule is
    # not deepcopy-able. Upstream sidesteps this by only ever copying `mlp` and
    # `obs_normalizer`; we need the encoders too, so detach the distribution for the
    # duration of the copy. The exported module never calls it -- it goes through
    # `as_deterministic_output_module()` instead.
    detached = model._modules.pop("distribution", None)
    try:
      self.core = copy.deepcopy(model)
    finally:
      if detached is not None:
        model._modules["distribution"] = detached
    self.core._modules["distribution"] = None
    self.core.eval()
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self._proprio_dim = model.history_length * sum(model.proprio_term_dims)
    self._action_dim = model.history_length * model.action_dim
    self._command_dim = model.num_command_tokens * model.command_token_dim

  def forward(
    self,
    proprio_hist: torch.Tensor,
    action_hist: torch.Tensor,
    command_window: torch.Tensor,
  ) -> torch.Tensor:
    obs = TensorDict(
      {
        PROPRIO_HIST: proprio_hist,
        ACTION_HIST: action_hist,
        COMMAND_WINDOW: command_window,
      },
      batch_size=[proprio_hist.shape[0]],
    )
    latent = self.core.get_latent(obs)
    return self.deterministic_output(self.core.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self._proprio_dim),
      torch.zeros(1, self._action_dim),
      torch.zeros(1, self._command_dim),
    )

  @property
  def input_names(self) -> list[str]:
    return [PROPRIO_HIST, ACTION_HIST, COMMAND_WINDOW]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass


def sinusoidal_positional_encoding(length: int, dim: int) -> torch.Tensor:
  """Fixed sinusoidal encoding (RGMT Eq. 9 / Eq. 14)."""
  if length <= 0:
    raise ValueError(f"length must be positive, got {length}.")
  if dim <= 0:
    raise ValueError(f"dim must be positive, got {dim}.")
  pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
  div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
  pe = torch.zeros(length, dim)
  pe[:, 0::2] = torch.sin(pos * div)
  # For odd dimensions the cosine branch has one fewer column.
  pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
  return pe.unsqueeze(0)
