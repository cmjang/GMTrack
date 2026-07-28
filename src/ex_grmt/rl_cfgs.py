"""RL runner configurations (paper Table III).

Two stages plus the ablation variants used in Fig. 7/9 and Table VIII. All PPO
hyper-parameters are shared; the stages differ only in the PACE/STAR block.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg

from ex_grmt.rsl_rl.config import (
  ExGRMTActorCfg,
  ExGRMTRunnerCfg,
  PacePpoAlgorithmCfg,
)

# Table III, PPO optimization.
_PPO = dict(
  num_learning_epochs=5,
  num_mini_batches=4,
  learning_rate=1.0e-3,
  schedule="adaptive",
  desired_kl=0.01,
  gamma=0.99,
  lam=0.95,
  clip_param=0.2,
  entropy_coef=0.005,
  value_loss_coef=1.0,
  use_clipped_value_loss=True,
  max_grad_norm=1.0,
)

# Table III, PACE.
ACQUISITION_FRACTION = 0.8
LAMBDA_BASE = 0.3
KAPPA = 5.0
RHO_REF = 0.6
BETA = 0.99

# Sec. V-B, STAR.
RHO_TOPK = 0.05
RHO_STAR = 0.25


def _actor(**overrides) -> ExGRMTActorCfg:
  return ExGRMTActorCfg(
    hidden_dims=(1024, 1024, 512, 256),
    activation="elu",
    obs_normalization=False,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    **overrides,
  )


def _critic() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(1024, 1024, 512, 512),
    activation="elu",
    obs_normalization=True,
  )


def stage1_runner_cfg() -> ExGRMTRunnerCfg:
  """Stage I: generalist base policy ``pi_base`` over the full motion distribution.

  No PACE split and no STAR -- Stage I is plain PPO with adaptive bin sampling
  (Sec. IV), so ``PacePPO`` degrades to upstream ``PPO`` behaviour here.
  """
  return ExGRMTRunnerCfg(
    actor=_actor(),
    critic=_critic(),
    algorithm=PacePpoAlgorithmCfg(
      acquisition_fraction=None,
      consolidation_enabled=False,
      use_star=False,
      **_PPO,
    ),
    experiment_name="ex_grmt_stage1",
    num_steps_per_env=24,
    max_iterations=30_000,
    save_interval=500,
    logger="tensorboard",
  )


def stage2_runner_cfg(
  base_checkpoint: str | None = None,
  use_star: bool = True,
  consolidation_enabled: bool = True,
  fixed_lambda_con: float | None = None,
  experiment_name: str = "ex_grmt_stage2",
) -> ExGRMTRunnerCfg:
  """Stage II: PACE + STAR expansion toward highly dynamic skills.

  Args:
    base_checkpoint: Stage-I checkpoint. Required at run time -- pass it on the CLI
      with ``--agent.algorithm.base-checkpoint`` if it is not baked in here.
    use_star: False reproduces the "w/o STAR" ablation.
    consolidation_enabled: False reproduces the "w/o L_con" ablation.
    fixed_lambda_con: Set to 0.5 for the "Fixed lambda_con" ablation.
  """
  return ExGRMTRunnerCfg(
    actor=_actor(),
    critic=_critic(),
    algorithm=PacePpoAlgorithmCfg(
      acquisition_fraction=ACQUISITION_FRACTION,
      lambda_base=LAMBDA_BASE,
      kappa=KAPPA,
      rho_ref=RHO_REF,
      beta=BETA,
      fixed_lambda_con=fixed_lambda_con,
      consolidation_enabled=consolidation_enabled,
      use_star=use_star,
      rho_topk=RHO_TOPK,
      rho_star=RHO_STAR,
      base_checkpoint=base_checkpoint,
      **_PPO,
    ),
    experiment_name=experiment_name,
    num_steps_per_env=24,
    max_iterations=50_000,
    save_interval=500,
    logger="tensorboard",
  )


def finetune_runner_cfg(base_checkpoint: str | None = None) -> ExGRMTRunnerCfg:
  """Baseline: naive fine-tuning on the challenging set (paper Fig. 5/7).

  Same warm start as Stage II, but every environment is an acquisition environment
  and there is no consolidation constraint -- this is the variant whose Generalist
  performance collapses in Fig. 7.
  """
  return ExGRMTRunnerCfg(
    actor=_actor(),
    critic=_critic(),
    algorithm=PacePpoAlgorithmCfg(
      acquisition_fraction=None,
      consolidation_enabled=False,
      use_star=False,
      base_checkpoint=base_checkpoint,
      **_PPO,
    ),
    experiment_name="ex_grmt_finetune",
    num_steps_per_env=24,
    max_iterations=50_000,
    save_interval=500,
    logger="tensorboard",
  )


def unified_encoder_runner_cfg(base_checkpoint: str | None = None) -> ExGRMTRunnerCfg:
  """Ablation "Unified Enc." (Table VIII)."""
  cfg = stage2_runner_cfg(base_checkpoint, experiment_name="ex_grmt_unified_enc")
  cfg.actor = _actor(unified_encoder=True)
  return cfg


def no_fsq_runner_cfg(base_checkpoint: str | None = None) -> ExGRMTRunnerCfg:
  """Ablation "w/o FSQ" (Table VIII)."""
  cfg = stage2_runner_cfg(base_checkpoint, experiment_name="ex_grmt_no_fsq")
  cfg.actor = _actor(use_fsq=False)
  return cfg
