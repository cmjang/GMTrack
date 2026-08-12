"""Config dataclasses extending mjlab's rsl-rl wrappers.

mjlab converts these to the raw ``train_cfg`` dict with ``dataclasses.asdict`` and
rsl-rl splats them into the constructors, so any field added here shows up as a
keyword argument of the corresponding class. mjlab also builds its tyro CLI from the
*annotations*, which is why the runner cfg re-annotates ``actor`` -- inheriting the
base annotation would hide every Extreme-RGMT knob from ``uv run train``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from ex_grmt.rsl_rl.fsq import SONIC_PROXY_FSQ_LEVELS


@dataclass
class ExGRMTActorCfg(RslRlModelCfg):
  """Actor config for :class:`~ex_grmt.rsl_rl.models.ExGRMTActor`."""

  class_name: str = "ex_grmt.rsl_rl.models:ExGRMTActor"
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

  class_name: str = "ex_grmt.rsl_rl.ppo_pace:PacePPO"

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


@dataclass
class ExGRMTRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config wiring the Extreme-RGMT actor, critic and algorithm together."""

  class_name: str = "OnPolicyRunner"
  actor: ExGRMTActorCfg = field(default_factory=ExGRMTActorCfg)
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
      # Order matters: ExGRMTActor derives branch dimensions positionally.
      "actor": ("proprio_hist", "action_hist", "command_window"),
      "critic": ("critic",),
      # NOTE: the "star" observation group is deliberately absent. It rides along in
      # the rollout storage for STAR and must not reach any network.
    }
  )
