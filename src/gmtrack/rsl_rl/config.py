"""Config dataclasses extending mjlab's rsl-rl wrappers.

mjlab converts these to the raw ``train_cfg`` dict with ``dataclasses.asdict`` and
rsl-rl splats them into the constructors, so any field added here shows up as a
keyword argument of the corresponding class. mjlab also builds its tyro CLI from the
*annotations*, which is why the runner cfg re-annotates ``actor`` -- inheriting the
base annotation would hide every GMTrack knob from ``uv run train``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from gmtrack.rsl_rl.fsq import SONIC_PROXY_FSQ_LEVELS


@dataclass
class GMTrackActorCfg(RslRlModelCfg):
  """Actor config for :class:`~gmtrack.rsl_rl.models.GMTrackActor`."""

  class_name: str = "gmtrack.rsl_rl.models:GMTrackActor"
  hidden_dims: Tuple[int, ...] = (1024, 1024, 512, 256)
  """Actor trunk (paper Table III)."""
  obs_normalization: bool = False
  """Must stay False: per-branch LayerNorm replaces running-statistic normalization."""

  history_length: int = 10
  """``H``. Paper uses a 10-frame proprioceptive and action history."""
  proprio_term_dims: Tuple[int, ...] = (3, 3, 29, 29)
  """Per-term widths in the ``proprio_hist`` group: gravity, ang vel, joint pos, joint vel."""
  token_dim: int = 64
  """Width of every branch embedding and of ``u_t``."""
  state_encoder_hidden: Tuple[int, ...] = (128,)
  """``f_o`` hidden widths. Table III lists the encoder as [64, 128, 64]."""
  action_encoder_hidden: Tuple[int, ...] = (64,)
  """``f_a`` hidden widths. Table III: [29, 64, 64]."""
  command_token_dim: int = 38
  """Per-reference-token width: 38 normally, 44 with heading closed-loop input."""
  command_encoder_hidden: Tuple[int, ...] = (128,)
  """``f_g`` hidden widths after the configurable command-token input."""
  use_command_valid_mask: bool = False
  """Mask boundary-clamped command tokens in the causal actor's cross-attention."""
  use_history_valid_mask: bool = False
  """Mask synthetic proprio/action history slots after episode reset."""
  command_window_offsets: Tuple[int, ...] | None = None
  """Exact actor reference offsets used by offset-aware sinusoidal encoding.

  ``None`` preserves the legacy slot-index encoding bit-for-bit. Causal tasks must
  provide the exact non-uniform layout so timing cannot be inferred from slot rank.
  """
  use_intent_aux: bool = False
  """Enable the training-only stochastic intent/future reconstruction branch."""
  use_intent_in_actor: bool = False
  """Append the causal intent posterior mean to the deployed actor input.

  This remains zero-lookahead: the mean is predicted only from the actor's causal
  representation.  Future tokens supervise it during training but are never policy
  inputs.
  """
  intent_latent_dim: int = 64
  """Gaussian intent width; 64 follows DAJI (arXiv:2605.14417)."""
  future_reconstruction_offsets: Tuple[int, ...] = ()
  """Sparse prediction horizons; causal tasks use TeleGate's (+5,+10,+20)."""
  intent_hidden_dims: Tuple[int, ...] = (128,)
  """Local posterior/decoder MLP widths; configurable engineering choice."""
  encoder_activation: str = "elu"
  num_heads: int = 4
  """ASSUMPTION: attention head count is not stated in the paper."""
  use_fsq: bool = True
  """False reproduces the "w/o FSQ" ablation (Table VIII)."""
  fsq_levels: int = SONIC_PROXY_FSQ_LEVELS
  """ASSUMPTION: paper omits this; 32 follows SONIC's matching 2 x 32 tokenizer."""
  fsq_token_dim: int = 32
  """Paper: ``u_t`` is factorized into two 32-dimensional tokens."""
  unified_encoder: bool = False
  """True reproduces the "Unified Enc." ablation (Table VIII)."""


@dataclass
class PacePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config extended with PACE (Sec. V-A) and STAR (Sec. V-B) parameters."""

  class_name: str = "gmtrack.rsl_rl.ppo_pace:PacePPO"

  acquisition_fraction: float | None = None
  """``xi``. None disables the PACE split (Stage I). Paper Table III: 0.8."""
  lambda_base: float = 0.3
  kappa: float = 5.0
  rho_ref: float = 0.6
  beta: float = 0.99
  fixed_lambda_con: float | None = None
  """Set (e.g. 0.5) for the "Fixed lambda_con" ablation (Fig. 9)."""
  consolidation_enabled: bool = True
  """False reproduces the "w/o L_con" ablation (Fig. 9)."""

  use_star: bool = True
  """False reproduces the "w/o STAR" ablation (Fig. 7, Table VII)."""
  rho_topk: float = 0.05
  """Fragment retention ratio per difficulty bin (Eq. 26)."""
  rho_star: float = 0.25
  """Share of each acquisition mini-batch drawn from the STAR pool (Eq. 28)."""

  base_checkpoint: str | None = None
  """Stage-I checkpoint. Supplies both the pi_theta warm start and the frozen pi_ref."""
  intent_reconstruction_coef: float = 0.0
  """Future reconstruction weight; TeleGate publishes 0.5."""
  intent_kl_coef: float = 0.0
  """Gaussian KL weight; TeleGate publishes 0.0005."""


@dataclass
class GMTrackRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config wiring the GMTrack actor, critic and algorithm together."""

  class_name: str = "OnPolicyRunner"
  actor: GMTrackActorCfg = field(default_factory=GMTrackActorCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(1024, 1024, 512, 512),
      activation="elu",
      obs_normalization=True,
    )
  )
  algorithm: PacePpoAlgorithmCfg = field(default_factory=PacePpoAlgorithmCfg)
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      # Order matters: GMTrackActor derives branch dimensions positionally.
      "actor": ("proprio_hist", "action_hist", "command_window"),
      "critic": ("critic",),
      # NOTE: the "star" observation group is deliberately absent. It rides along in
      # the rollout storage for STAR and must not reach any network.
    }
  )
