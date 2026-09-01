"""RL runner configurations (paper Table III).

Two stages plus the ablation variants used in Fig. 7/9 and Table VIII. All PPO
hyper-parameters are shared; the stages differ only in the PACE/STAR block.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg

from gmtrack.envs.env_cfg import (
  CAUSAL_ACTOR_WINDOW_OFFSETS,
  CAUSAL_RECONSTRUCTION_OFFSETS,
)
from gmtrack.rsl_rl.config import (
  GMTrackActorCfg,
  GMTrackRunnerCfg,
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

# TeleGate (arXiv:2602.09628) public auxiliary-objective weights.
INTENT_RECONSTRUCTION_COEF = 0.5
INTENT_KL_COEF = 0.0005


def _actor(**overrides) -> GMTrackActorCfg:
  return GMTrackActorCfg(
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


def _with_causal_observation_groups(
  cfg: GMTrackRunnerCfg,
  causal_online: bool,
  actor_boundary_masks: bool = True,
) -> GMTrackRunnerCfg:
  """Install the causal actor and history-conditioned privileged critic ABI.

  ``actor_boundary_masks=False`` is the ablation of the boundary bookkeeping: the
  actor stops being told which of its tokens are clamped padding, matching the
  behaviour before validity masks existed. The groups themselves stay in the
  environment because the critic still consumes ``history_valid_mask``, and the
  reconstruction head still masks its targets -- feeding that head fabricated frames
  would be a defect rather than an ablation.
  """
  if causal_online:
    cfg.obs_groups["actor"] = (
      "proprio_hist",
      "action_hist",
      "command_window",
    ) + (("history_valid_mask", "past_valid_mask") if actor_boundary_masks else ())
    # V(h, s, g_future, mask): retaining actor-visible history avoids the biased
    # state-only critic for a history-dependent policy described by Baisero & Amato
    # (arXiv:2105.11674) and matches the actor-observation-plus-privilege form in
    # Informed Asymmetric Actor-Critic (arXiv:2509.26000). GigaBrain-WBC-0.5
    # (arXiv:2608.18234) provides the concrete humanoid precedent: future reference
    # plus 10-frame proprio/action history. Future reference is training-only.
    cfg.obs_groups["critic"] = (
      "proprio_hist",
      "action_hist",
      "history_valid_mask",
      "critic",
      "command_future_window",
      "future_valid_mask",
    )
  return cfg


def stage1_runner_cfg(
  *,
  heading_closed_loop: bool = False,
  experiment_name: str | None = None,
  causal_online: bool = False,
  actor_boundary_masks: bool = True,
  intent_in_actor: bool = False,
) -> GMTrackRunnerCfg:
  """Stage I: generalist base policy ``pi_base`` over the full motion distribution.

  No PACE split and no STAR -- Stage I is plain PPO with adaptive bin sampling
  (Sec. IV), so ``PacePPO`` degrades to upstream ``PPO`` behaviour here.

  Args:
    heading_closed_loop: Append the 6D root-orientation error to every command token.
    causal_online: Enable the offset-aware sparse causal actor, intent auxiliary
      objective, and history-conditioned future-privileged critic.
    actor_boundary_masks: Give the actor its past/history validity masks. False is
      the ablation: the same past-only window, but the actor is not told which
      tokens are boundary padding. Critic and reconstruction masking are unchanged.
    intent_in_actor: Append the causal intent posterior mean to the policy trunk.
      This requires ``causal_online=True`` and remains zero-lookahead at deployment.
  """
  if intent_in_actor and not causal_online:
    raise ValueError("intent_in_actor=True requires causal_online=True.")
  return _with_causal_observation_groups(
    GMTrackRunnerCfg(
      actor=_actor(
        command_token_dim=44 if heading_closed_loop else 38,
        use_command_valid_mask=causal_online and actor_boundary_masks,
        use_history_valid_mask=causal_online and actor_boundary_masks,
        command_window_offsets=(CAUSAL_ACTOR_WINDOW_OFFSETS if causal_online else None),
        use_intent_aux=causal_online,
        use_intent_in_actor=intent_in_actor,
        intent_latent_dim=64,
        future_reconstruction_offsets=(
          CAUSAL_RECONSTRUCTION_OFFSETS if causal_online else ()
        ),
      ),
      critic=_critic(),
      algorithm=PacePpoAlgorithmCfg(
        acquisition_fraction=None,
        consolidation_enabled=False,
        use_star=False,
        intent_reconstruction_coef=(
          INTENT_RECONSTRUCTION_COEF if causal_online else 0.0
        ),
        intent_kl_coef=INTENT_KL_COEF if causal_online else 0.0,
        **_PPO,
      ),
      experiment_name=(
        experiment_name
        if experiment_name is not None
        else "gmtrack_stage1_causal_intent_actor"
        if causal_online and intent_in_actor
        else "gmtrack_stage1_causal_heading"
        if causal_online and heading_closed_loop
        else "gmtrack_stage1_causal"
        if causal_online
        else "gmtrack_stage1_heading"
        if heading_closed_loop
        else "gmtrack_stage1"
      ),
      num_steps_per_env=24,
      # Paper gives no iteration count. BeyondMimic ships 30k and SONIC 100k; we take
      # 100k -- 30k gets a plain tracker somewhere, but fall-recovery training needs
      # considerably longer. RECOVERY_ASSIST_ANNEAL_STEPS is 100k x 24 to match.
      max_iterations=100_000,
      save_interval=500,
      logger="tensorboard",
    ),
    causal_online,
    actor_boundary_masks,
  )


def stage2_runner_cfg(
  base_checkpoint: str | None = None,
  use_star: bool = True,
  consolidation_enabled: bool = True,
  fixed_lambda_con: float | None = None,
  experiment_name: str | None = None,
  heading_closed_loop: bool = False,
  causal_online: bool = False,
  intent_in_actor: bool = False,
) -> GMTrackRunnerCfg:
  """Stage II: PACE + STAR expansion toward highly dynamic skills.

  Args:
    base_checkpoint: Stage-I checkpoint. Required at run time -- pass it on the CLI
      with ``--agent.algorithm.base-checkpoint`` if it is not baked in here.
    use_star: False reproduces the "w/o STAR" ablation.
    consolidation_enabled: False reproduces the "w/o L_con" ablation.
    fixed_lambda_con: Set to 0.5 for the "Fixed lambda_con" ablation.
    experiment_name: Optional explicit run family. When omitted, heading and
      open-loop policies receive distinct default names.
    heading_closed_loop: Append the 6D root-orientation error to every command token.
    causal_online: Enable the offset-aware sparse causal actor, intent auxiliary
      objective, and history-conditioned future-privileged critic.
    intent_in_actor: Append the causal intent posterior mean to the policy trunk.
      This requires ``causal_online=True`` and must match the Stage-I checkpoint.
  """
  if intent_in_actor and not causal_online:
    raise ValueError("intent_in_actor=True requires causal_online=True.")
  return _with_causal_observation_groups(
    GMTrackRunnerCfg(
      actor=_actor(
        command_token_dim=44 if heading_closed_loop else 38,
        use_command_valid_mask=causal_online,
        use_history_valid_mask=causal_online,
        command_window_offsets=(CAUSAL_ACTOR_WINDOW_OFFSETS if causal_online else None),
        use_intent_aux=causal_online,
        use_intent_in_actor=intent_in_actor,
        intent_latent_dim=64,
        future_reconstruction_offsets=(
          CAUSAL_RECONSTRUCTION_OFFSETS if causal_online else ()
        ),
      ),
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
        intent_reconstruction_coef=(
          INTENT_RECONSTRUCTION_COEF if causal_online else 0.0
        ),
        intent_kl_coef=INTENT_KL_COEF if causal_online else 0.0,
        **_PPO,
      ),
      experiment_name=(
        experiment_name
        if experiment_name is not None
        else "gmtrack_stage2_causal_intent_actor"
        if causal_online and intent_in_actor
        else "gmtrack_stage2_causal_heading"
        if causal_online and heading_closed_loop
        else "gmtrack_stage2_causal"
        if causal_online
        else "gmtrack_stage2_heading"
        if heading_closed_loop
        else "gmtrack_stage2"
      ),
      num_steps_per_env=24,
      max_iterations=50_000,
      save_interval=500,
      logger="tensorboard",
    ),
    causal_online,
  )


def finetune_runner_cfg(base_checkpoint: str | None = None) -> GMTrackRunnerCfg:
  """Baseline: naive fine-tuning on the challenging set (paper Fig. 5/7).

  Same warm start as Stage II, but every environment is an acquisition environment
  and there is no consolidation constraint -- this is the variant whose Generalist
  performance collapses in Fig. 7.
  """
  return GMTrackRunnerCfg(
    actor=_actor(),
    critic=_critic(),
    algorithm=PacePpoAlgorithmCfg(
      acquisition_fraction=None,
      consolidation_enabled=False,
      use_star=False,
      base_checkpoint=base_checkpoint,
      **_PPO,
    ),
    experiment_name="gmtrack_finetune",
    num_steps_per_env=24,
    max_iterations=50_000,
    save_interval=500,
    logger="tensorboard",
  )


def unified_encoder_runner_cfg(base_checkpoint: str | None = None) -> GMTrackRunnerCfg:
  """Ablation "Unified Enc." (Table VIII)."""
  cfg = stage2_runner_cfg(base_checkpoint, experiment_name="gmtrack_unified_enc")
  cfg.actor = _actor(unified_encoder=True)
  return cfg


def no_fsq_runner_cfg(base_checkpoint: str | None = None) -> GMTrackRunnerCfg:
  """Ablation "w/o FSQ" (Table VIII)."""
  cfg = stage2_runner_cfg(base_checkpoint, experiment_name="gmtrack_no_fsq")
  cfg.actor = _actor(use_fsq=False)
  return cfg
