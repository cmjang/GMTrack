"""PACE: Progressive Acquisition and Consolidation for Expansion (paper Sec. V-A).

Stage II optimizes a single augmented policy against two asymmetric objectives
(Eq. 14)::

    min_theta  L_Acquisition^{D_c}  +  lambda_con^t * L_Consolidation^{D_m}

Acquisition transitions come from the ``xi`` fraction of environments that sample the
challenging set adaptively; consolidation transitions come from the remainder, which
sample the mastered set uniformly. Only acquisition rows drive the PPO surrogate;
only consolidation rows drive the behaviour-cloning term against the frozen Stage-I
policy ``pi_ref`` (Eq. 15).

The consolidation weight follows training progress (Eq. 17-19): as the acquisition
environments start surviving long enough to produce valid samples, the constraint
toward ``pi_ref`` is tightened so specialist learning does not drift the policy off
the mastered repertoire.

IMPLEMENTATION DETAILS:

* *Valid sample* = a rollout step at or before the environment's first true tracking
  failure. Timeouts and ordinary motion ends do not shorten this prefix. See
  :meth:`~ex_grmt.rsl_rl.storage.StarRolloutStorage.valid_sample_counts`.
* Acquisition rows alone drive advantage normalization, PPO surrogate, value,
  entropy, adaptive-KL, and the causal task's intent reconstruction/KL auxiliary
  objective. Consolidation rows are reserved exclusively for Eq. (15), matching the
  PACE rule that each rollout group contributes only its corresponding loss.
* Distributed runs aggregate the sufficient statistics used by PACE and STAR before
  updating them, so every rank applies the same normalization and consolidation
  weight.
"""

from __future__ import annotations

import copy
from typing import cast

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from ex_grmt.mdp.commands import MultiMotionCommand
from ex_grmt.provenance import (
  CHECKPOINT_OBSERVATION_SCHEMA_KEY,
  build_observation_schema,
  validate_checkpoint_observation_schema,
)
from ex_grmt.rsl_rl.models import ExGRMTActor
from ex_grmt.rsl_rl.storage import TRACKING_FAILURES_EXTRA, StarRolloutStorage


def _runtime_observation_schema(
  obs: TensorDict,
  env: VecEnv,
  actor_cfg: dict,
  obs_groups: dict[str, list[str]],
) -> dict:
  """Describe the exact actor/critic observation ABI constructed for this run."""
  raw_env = env.unwrapped
  command = raw_env.command_manager.get_term("motion")
  if not isinstance(command, MultiMotionCommand):
    raise TypeError("ExGRMT observation schema requires MultiMotionCommand.")

  command_token_dim = int(actor_cfg["command_token_dim"])
  if command_token_dim != command.command_token_dim:
    raise ValueError(
      "Actor and command token widths differ while building observation schema: "
      f"{command_token_dim} != {command.command_token_dim}."
    )
  actor_groups = list(obs_groups["actor"])
  critic_groups = list(obs_groups["critic"])
  use_intent_aux = bool(actor_cfg["use_intent_aux"])
  schema_only_groups = (
    ["future_reconstruction_target", "future_reconstruction_valid_mask"]
    if use_intent_aux
    else []
  )
  group_widths = {
    name: int(obs[name].shape[-1])
    for name in dict.fromkeys((*actor_groups, *critic_groups, *schema_only_groups))
  }
  configured_actor_offsets = actor_cfg["command_window_offsets"]
  runtime_actor_offsets = command.window_offsets.detach().cpu().tolist()
  command_has_explicit_offsets = command.cfg.command_window_offsets is not None
  actor_has_explicit_offsets = configured_actor_offsets is not None
  if command_has_explicit_offsets != actor_has_explicit_offsets:
    raise ValueError(
      "Actor and command must either both configure exact window offsets or both "
      "use the legacy radius/slot-index contract."
    )
  if configured_actor_offsets is not None and list(configured_actor_offsets) != runtime_actor_offsets:
    raise ValueError(
      "Actor positional-encoding offsets do not match the command window: "
      f"{list(configured_actor_offsets)} != {runtime_actor_offsets}."
    )
  configured_reconstruction_offsets = list(
    actor_cfg["future_reconstruction_offsets"]
  )
  runtime_reconstruction_offsets = (
    command.reconstruction_window_offsets.detach().cpu().tolist()
  )
  if configured_reconstruction_offsets != runtime_reconstruction_offsets:
    raise ValueError(
      "Actor reconstruction offsets do not match the command targets: "
      f"{configured_reconstruction_offsets} != {runtime_reconstruction_offsets}."
    )
  critic_offsets = command.critic_window_offsets.detach().cpu().tolist()
  use_history_valid_mask = bool(actor_cfg["use_history_valid_mask"])
  use_past_valid_mask = bool(actor_cfg["use_command_valid_mask"])
  use_future_valid_mask = bool(critic_offsets)
  expected_future_groups = {"command_future_window", "future_valid_mask"}
  if use_future_valid_mask != expected_future_groups.issubset(critic_groups):
    raise ValueError(
      "Critic future offsets and future observation groups must be enabled together."
    )
  return build_observation_schema(
    actor_observation_groups=actor_groups,
    critic_observation_groups=critic_groups,
    observation_group_widths=group_widths,
    command_window_offsets=runtime_actor_offsets,
    critic_window_offsets=critic_offsets,
    reconstruction_window_offsets=runtime_reconstruction_offsets,
    command_fps=command.lib.fps,
    command_token_dim=command_token_dim,
    heading_closed_loop=bool(command.cfg.heading_closed_loop),
    history_length=int(actor_cfg["history_length"]),
    proprio_term_names=list(
      raw_env.observation_manager.active_terms["proprio_hist"]
    ),
    proprio_term_dims=list(actor_cfg["proprio_term_dims"]),
    action_dim=int(env.num_actions),
    use_past_valid_mask=use_past_valid_mask,
    use_history_valid_mask=use_history_valid_mask,
    use_future_valid_mask=use_future_valid_mask,
    use_reconstruction_valid_mask=use_intent_aux,
    command_position_encoding=(
      "legacy_sinusoidal_slot_index"
      if configured_actor_offsets is None
      else "sinusoidal_normalized_actual_offset"
    ),
    use_intent_aux=use_intent_aux,
    intent_latent_dim=int(actor_cfg["intent_latent_dim"]),
  )


class PacePPO(PPO):
  """PPO with PACE consolidation and STAR mini-batch resampling."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: StarRolloutStorage,
    observation_schema: dict,
    require_observation_schema: bool,
    acquisition_fraction: float | None = None,
    lambda_base: float = 0.3,
    kappa: float = 5.0,
    rho_ref: float = 0.6,
    beta: float = 0.99,
    fixed_lambda_con: float | None = None,
    consolidation_enabled: bool = True,
    use_star: bool = True,
    rho_topk: float = 0.05,
    rho_star: float = 0.25,
    base_checkpoint: str | None = None,
    intent_reconstruction_coef: float = 0.0,
    intent_kl_coef: float = 0.0,
    **kwargs,
  ) -> None:
    unsupported = []
    if actor.is_recurrent or critic.is_recurrent:
      unsupported.append("recurrent actor/critic")
    if kwargs.get("rnd_cfg"):
      unsupported.append("RND")
    if kwargs.get("symmetry_cfg"):
      unsupported.append("symmetry")
    if unsupported:
      raise ValueError(
        "PacePPO.update does not implement "
        + ", ".join(unsupported)
        + ". Disable these extensions or use upstream PPO."
      )

    super().__init__(actor, critic, storage, **kwargs)
    self.storage: StarRolloutStorage = storage

    self.acquisition_fraction = acquisition_fraction
    self.lambda_base = lambda_base
    self.kappa = kappa
    self.rho_ref = rho_ref
    self.beta = beta
    self.fixed_lambda_con = fixed_lambda_con
    self.consolidation_enabled = (
      consolidation_enabled and acquisition_fraction is not None
    )
    self.use_star = use_star
    # Stored for provenance/logging; the storage owns the actual resampling knobs.
    self.rho_topk = rho_topk
    self.rho_star = rho_star
    self.base_checkpoint = base_checkpoint
    if intent_reconstruction_coef < 0.0 or intent_kl_coef < 0.0:
      raise ValueError("Intent reconstruction and KL coefficients must be non-negative.")
    self.intent_reconstruction_coef = intent_reconstruction_coef
    self.intent_kl_coef = intent_kl_coef
    self.use_intent_aux = (
      intent_reconstruction_coef > 0.0 or intent_kl_coef > 0.0
    )
    if self.use_intent_aux:
      if not isinstance(self._raw_actor, ExGRMTActor):
        raise TypeError("Intent auxiliary losses require ExGRMTActor.")
      if not self._raw_actor.use_intent_aux:
        raise ValueError(
          "Nonzero intent loss coefficients require actor.use_intent_aux=True."
        )
    self.observation_schema = observation_schema
    self.require_observation_schema = require_observation_schema

    # Eq. (18): the smoothed ratio starts at rho_ref so lambda_con starts at lambda_base.
    self.rho_bar = rho_ref
    self.lambda_con = lambda_base if fixed_lambda_con is None else fixed_lambda_con

    self.actor_ref: MLPModel | None = None
    self.initial_reference_diagnostics: dict[str, float] | None = None
    self.last_valid_sample_diagnostics: dict[str, int | str] = {}

  # -- reference policy -----------------------------------------------------

  def attach_reference_policy(self, actor_ref: MLPModel) -> None:
    """Install the frozen Stage-I policy used by the consolidation loss.

    Kept out of ``self.optimizer`` (built in ``PPO.__init__`` from actor+critic only).
    :meth:`save` serializes it separately so Stage-II checkpoints remain self-contained.
    """
    actor_ref.eval()
    for p in actor_ref.parameters():
      p.requires_grad_(False)
    self.actor_ref = actor_ref

  @staticmethod
  def _deterministic_policy_mean(
    actor: MLPModel, observations: TensorDict
  ) -> torch.Tensor:
    """Return the policy's raw deterministic mean, never a sampled action."""
    return actor(observations, stochastic_output=False)

  @torch.no_grad()
  def reference_policy_diagnostics(self, observations: TensorDict) -> dict[str, float]:
    """Compare current and reference policy means/distributions on observations."""
    if self.actor_ref is None:
      raise ValueError("No reference policy is attached.")

    parameter = next(self.actor.parameters(), None)
    cuda_devices = (
      [parameter.device.index]
      if parameter is not None
      and parameter.is_cuda
      and parameter.device.index is not None
      else []
    )
    # Stochastic forwards are needed only to populate rsl-rl's distribution objects
    # for KL. Preserve RNG state so this assertion/diagnostic cannot alter training.
    with torch.random.fork_rng(devices=cuda_devices):
      current_mean = self._deterministic_policy_mean(self.actor, observations)
      reference_mean = self._deterministic_policy_mean(self.actor_ref, observations)
      self.actor(observations, stochastic_output=True)
      current_params = tuple(
        p.detach().clone() for p in self.actor.output_distribution_params
      )
      self.actor_ref(observations, stochastic_output=True)
      reference_params = tuple(
        p.detach().clone() for p in self.actor_ref.output_distribution_params
      )
      kl = self.actor.get_kl_divergence(reference_params, current_params)

    difference = current_mean - reference_mean
    return {
      "deterministic_mean_max_abs_diff": float(difference.abs().max().item()),
      "deterministic_mean_mse": float(difference.square().mean().item()),
      "reference_kl_mean": float(kl.mean().item()),
      "reference_kl_max": float(kl.max().item()),
    }

  def assert_reference_policy_initialized(
    self, observations: TensorDict, *, atol: float = 1.0e-7
  ) -> dict[str, float]:
    """Assert Algorithm 1 starts with ``pi_theta == pi_ref`` on a probe batch."""
    diagnostics = self.reference_policy_diagnostics(observations)
    if (
      diagnostics["deterministic_mean_max_abs_diff"] > atol
      or diagnostics["reference_kl_max"] > atol
    ):
      raise RuntimeError(
        "PACE initialization requires current actor and actor_ref to match; got "
        f"mean max diff={diagnostics['deterministic_mean_max_abs_diff']:.3e}, "
        f"KL max={diagnostics['reference_kl_max']:.3e}."
      )
    self.initial_reference_diagnostics = diagnostics
    return diagnostics

  def train_mode(self) -> None:
    super().train_mode()
    if self.consolidation_enabled and self.actor_ref is None:
      raise ValueError(
        "PACE consolidation training needs a frozen reference policy. Supply a "
        "Stage-I base_checkpoint for a fresh run, or load a Stage-II checkpoint "
        "containing actor_ref_state_dict before calling learn()."
      )
    if self.actor_ref is not None:
      self.actor_ref.eval()  # Never leaves eval.

  # -- persistence ---------------------------------------------------------

  def save(self) -> dict:
    """Extend the upstream checkpoint with PACE and frozen-reference state."""
    saved = super().save()
    saved["pace_state_dict"] = {
      "version": 1,
      "rho_bar": self.rho_bar,
      "lambda_con": self.lambda_con,
      "learning_rate": self.learning_rate,
    }
    saved[CHECKPOINT_OBSERVATION_SCHEMA_KEY] = copy.deepcopy(
      self.observation_schema
    )
    if self.actor_ref is not None:
      saved["actor_ref_state_dict"] = self.actor_ref.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Restore PACE state and make optimizer-LR resume semantics explicit.

    ``load_cfg["optimizer_lr"]`` accepts ``"checkpoint"`` (the exact-resume
    default) or ``"config"`` (load optimizer moments but use the newly constructed
    run's configured learning rate). This also keeps ``self.learning_rate`` and the
    optimizer param groups from silently disagreeing.
    """
    validate_checkpoint_observation_schema(
      loaded_dict,
      self.observation_schema,
      require_present=self.require_observation_schema,
    )
    configured_lr = self.learning_rate
    load_optimizer = load_cfg is None or bool(load_cfg.get("optimizer", False))
    load_iteration = super().load(loaded_dict, load_cfg, strict)

    pace_state = loaded_dict.get("pace_state_dict")
    if pace_state is not None:
      if pace_state.get("version") != 1:
        raise ValueError(
          f"Unsupported PACE state version {pace_state.get('version')!r}."
        )
      self.rho_bar = float(pace_state.get("rho_bar", self.rho_bar))
      self.lambda_con = float(pace_state.get("lambda_con", self.lambda_con))

    actor_ref_state = loaded_dict.get("actor_ref_state_dict")
    if actor_ref_state is not None and self.consolidation_enabled:
      if self.actor_ref is None:
        self.attach_reference_policy(copy.deepcopy(self._raw_actor))
      self.actor_ref.load_state_dict(actor_ref_state, strict=strict)

    if load_optimizer:
      lr_policy = (
        "checkpoint" if load_cfg is None else load_cfg.get("optimizer_lr", "checkpoint")
      )
      if lr_policy == "checkpoint":
        group_lrs = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(group_lrs) != 1:
          raise ValueError(
            "PACE expects one optimizer learning rate, but checkpoint contains "
            f"{sorted(group_lrs)}."
          )
        self.learning_rate = group_lrs.pop()
      elif lr_policy == "config":
        self.learning_rate = configured_lr
        for group in self.optimizer.param_groups:
          group["lr"] = configured_lr
      else:
        raise ValueError(
          "load_cfg['optimizer_lr'] must be 'checkpoint' or 'config', got "
          f"{lr_policy!r}."
        )

    return load_iteration

  # -- returns --------------------------------------------------------------

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record true tracking failures separately from combined episode ``done``.

    An explicit terminated mask is lossless. ``done & ~time_out`` is retained as a
    compatibility path for stock rsl-rl wrappers, but diagnostics expose its use
    because it cannot represent simultaneous termination and truncation.
    """
    if TRACKING_FAILURES_EXTRA in extras:
      tracking_failures = extras[TRACKING_FAILURES_EXTRA]
      derived = False
    elif "time_outs" in extras:
      tracking_failures = (
        dones.reshape(-1).bool() & ~extras["time_outs"].reshape(-1).bool()
      )
      derived = True
    else:
      raise RuntimeError(
        "PACE needs a true tracking failure mask. Pass "
        f"extras[{TRACKING_FAILURES_EXTRA!r}] from the environment's terminated "
        "signal, or provide extras['time_outs'] for compatibility derivation."
      )
    tracking_failures = tracking_failures.reshape(-1).bool()
    combined_dones = dones.reshape(-1).bool()
    if tracking_failures.shape != combined_dones.shape:
      raise ValueError(
        "tracking failure and done masks must have the same number of environments; "
        f"got {tuple(tracking_failures.shape)} and {tuple(combined_dones.shape)}."
      )
    if bool((tracking_failures & ~combined_dones).any()):
      raise ValueError("A tracking failure must also be marked done on the same step.")
    self.storage.record_tracking_failures(tracking_failures, derived=derived)
    super().process_env_step(obs, rewards, dones, extras)

  def compute_returns(self, obs: TensorDict) -> None:
    """GAE, then STAR's difficulty-conditioned advantage normalization (Eq. 21-23)."""
    st = self.storage
    critic_hidden_state = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    self.critic.reset(hidden_state=critic_hidden_state)

    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = (
        last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      )
      next_is_not_terminal = 1.0 - st.dones[step].float()
      delta = (
        st.rewards[step]
        + next_is_not_terminal * self.gamma * next_values
        - st.values[step]
      )
      advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
      st.returns[step] = advantage + st.values[step]

    st.advantages = st.returns - st.values
    # Eq. (21): keep A^raw for fragment scoring, *before* any normalization.
    st.raw_advantages.copy_(st.advantages)

    if self.normalize_advantage_per_mini_batch:
      return
    if self.use_star:
      st.normalize_advantages_by_difficulty(distributed=self.is_multi_gpu)
    else:
      st.normalize_acquisition_advantages(distributed=self.is_multi_gpu)

  # -- consolidation weight -------------------------------------------------

  def _update_lambda_con(self) -> tuple[int, int]:
    """Eq. (17)-(19). Returns the raw valid-sample counts for logging."""
    n_acq, n_con = self.storage.valid_sample_counts()
    diagnostics = self.storage.valid_sample_diagnostics()
    if self.is_multi_gpu:
      counts = torch.tensor([n_acq, n_con], dtype=torch.long, device=self.device)
      torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
      n_acq, n_con = int(counts[0].item()), int(counts[1].item())
      numeric_keys = [
        key
        for key, value in diagnostics.items()
        if key != "mode" and isinstance(value, int)
      ]
      if numeric_keys:
        values = torch.tensor(
          [int(diagnostics[key]) for key in numeric_keys],
          dtype=torch.long,
          device=self.device,
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        diagnostics.update(
          {
            key: int(value.item())
            for key, value in zip(numeric_keys, values, strict=True)
          }
        )
    if diagnostics:
      diagnostics["valid_acq_samples"] = n_acq
      diagnostics["valid_con_samples"] = n_con
    self.last_valid_sample_diagnostics = diagnostics
    if not self.consolidation_enabled:
      return n_acq, n_con

    total = n_acq + n_con
    rho = n_acq / total if total > 0 else self.rho_ref
    self.rho_bar = self.beta * self.rho_bar + (1.0 - self.beta) * rho

    if self.fixed_lambda_con is not None:
      self.lambda_con = self.fixed_lambda_con
    else:
      self.lambda_con = min(
        1.0, self.lambda_base + self.kappa * max(0.0, self.rho_bar - self.rho_ref)
      )
    return n_acq, n_con

  # -- update ---------------------------------------------------------------

  @staticmethod
  def _masked_intent_losses(
    reconstruction_rows: torch.Tensor,
    kl_rows: torch.Tensor,
    valid_counts: torch.Tensor,
    acq_mask: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce auxiliary rows using the storage-provided PACE acquisition mask."""
    if acq_mask.dtype is not torch.bool:
      raise TypeError(f"acq_mask must be bool, got {acq_mask.dtype}.")
    expected = acq_mask.shape
    for name, values in (
      ("reconstruction_rows", reconstruction_rows),
      ("kl_rows", kl_rows),
      ("valid_counts", valid_counts),
    ):
      if values.shape != expected:
        raise ValueError(
          f"{name} must match acq_mask shape {expected}, got {values.shape}."
        )
    if not bool(acq_mask.any()):
      raise RuntimeError("Intent auxiliary objective received no acquisition rows.")
    reconstruction_mask = acq_mask & (valid_counts > 0)
    if not bool(reconstruction_mask.any()):
      raise RuntimeError(
        "Acquisition mini-batch has no valid future reconstruction target."
      )
    return (
      reconstruction_rows[reconstruction_mask].mean(),
      kl_rows[acq_mask].mean(),
      valid_counts[acq_mask].float().mean(),
    )

  def update(self) -> dict[str, float]:  # noqa: C901 - mirrors upstream structure
    n_acq, n_con = self._update_lambda_con()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_consolidation = 0.0
    mean_intent_reconstruction = 0.0
    mean_intent_kl = 0.0
    mean_intent_valid_tokens = 0.0
    # Table III trains with the adaptive-KL schedule, whose only observable effect is
    # the learning rate. When that rate sits on rsl-rl's 1e-5 floor the schedule is
    # asking for a smaller step than it can take, and the run looks identical whether
    # the KL is 0.021 or 2.1 -- two very different diseases. Log the driving quantity.
    mean_kl = 0.0
    kl_batches = 0
    actor_grad_norm_sum = 0.0
    critic_grad_norm_sum = 0.0
    actor_grad_norm_max = 0.0
    critic_grad_norm_max = 0.0

    if self.use_star or self.consolidation_enabled:
      generator = self.storage.star_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      acq_mask = getattr(batch, "acq_mask", None)
      if acq_mask is None:
        acq_mask = torch.ones(
          batch.observations.batch_size[0], dtype=torch.bool, device=self.device
        )
      con_mask = ~acq_mask
      has_con = self.consolidation_enabled and bool(con_mask.any())

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = self._normalize_masked(
            batch.advantages, acq_mask, distributed=self.is_multi_gpu
          )

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        mean_kl += self._adapt_learning_rate(
          batch.old_distribution_params, distribution_params, acq_mask
        )
        kl_batches += 1

      # -- acquisition: clipped PPO surrogate over E_A rows only (Eq. 16) --
      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      adv = torch.squeeze(batch.advantages)
      surrogate = -adv * ratio
      surrogate_clipped = -adv * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped)[acq_mask].mean()

      # -- acquisition: critic loss over E_A rows only --
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_error = torch.max(
          (values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)
        )
      else:
        value_error = (batch.returns - values).pow(2)
      value_loss = value_error.squeeze(-1)[acq_mask].mean()

      acquisition_entropy = entropy[acq_mask].mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * acquisition_entropy
      )

      # Training-only intent objective. `acq_mask` is emitted by the same
      # pace_env_split-backed storage path as PPO/PACE; never reconstruct or
      # regularize consolidation rows, whose sole objective is L_con.
      if self.use_intent_aux:
        if not isinstance(self._raw_actor, ExGRMTActor):
          raise TypeError("Intent auxiliary losses require ExGRMTActor.")
        reconstruction_rows, kl_rows, valid_counts = (
          self._raw_actor.auxiliary_future_losses(batch.observations)
        )
        reconstruction_loss, intent_kl_loss, valid_token_mean = (
          self._masked_intent_losses(
            reconstruction_rows, kl_rows, valid_counts, acq_mask
          )
        )
        loss = (
          loss
          + self.intent_reconstruction_coef * reconstruction_loss
          + self.intent_kl_coef * intent_kl_loss
        )
        mean_intent_reconstruction += reconstruction_loss.item()
        mean_intent_kl += intent_kl_loss.item()
        mean_intent_valid_tokens += float(valid_token_mean.item())

      # -- consolidation: BC against the frozen Stage-I policy (Eq. 15) --
      if has_con:
        assert self.actor_ref is not None, (
          "PACE consolidation is enabled but no reference policy is attached; "
          "pass `base_checkpoint` so construct_algorithm can build pi_ref."
        )
        con_obs = batch.observations[con_mask]
        with torch.no_grad():
          ref_actions = self._deterministic_policy_mean(self.actor_ref, con_obs)
        # The stochastic acquisition forward above has already populated the raw
        # distribution mean. Select its consolidation rows; never use the sampled
        # return value from that forward as the Eq. 15 target.
        cur_actions = self.actor.output_mean[con_mask]
        consolidation_loss = (cur_actions - ref_actions).pow(2).sum(-1).mean()
        loss = loss + self.lambda_con * consolidation_loss
        mean_consolidation += consolidation_loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      actor_grad_norm = float(
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm).item()
      )
      critic_grad_norm = float(
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm).item()
      )
      actor_grad_norm_sum += actor_grad_norm
      critic_grad_norm_sum += critic_grad_norm
      actor_grad_norm_max = max(actor_grad_norm_max, actor_grad_norm)
      critic_grad_norm_max = max(critic_grad_norm_max, critic_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += acquisition_entropy.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_dict = {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "consolidation": mean_consolidation / num_updates,
      "lambda_con": self.lambda_con,
      "rho_bar": self.rho_bar,
      "valid_acq_samples": float(n_acq),
      "valid_con_samples": float(n_con),
      "star_pool_size": float(self.storage.last_star_pool_size),
      "learning_rate": self.learning_rate,
      # Mean over mini-batches of the KL the schedule reacted to. Compare against
      # `desired_kl`: > 2x means the rate was cut this update, < 0.5x means raised.
      "kl": mean_kl / kl_batches if kl_batches else float("nan"),
      "actor_grad_norm_mean": actor_grad_norm_sum / num_updates,
      "actor_grad_norm_max": actor_grad_norm_max,
      "critic_grad_norm_mean": critic_grad_norm_sum / num_updates,
      "critic_grad_norm_max": critic_grad_norm_max,
    }
    if self.use_intent_aux:
      loss_dict.update(
        {
          "intent_reconstruction": mean_intent_reconstruction / num_updates,
          "intent_kl": mean_intent_kl / num_updates,
          "intent_valid_future_tokens": mean_intent_valid_tokens / num_updates,
        }
      )
    # Stable shorthand keys are the mini-batch means; explicit mean/max fields retain
    # enough detail to distinguish isolated spikes from a persistently large update.
    loss_dict["actor_grad_norm"] = loss_dict["actor_grad_norm_mean"]
    loss_dict["critic_grad_norm"] = loss_dict["critic_grad_norm_mean"]
    for key, value in self.last_valid_sample_diagnostics.items():
      if key not in {"mode", "valid_acq_samples", "valid_con_samples"}:
        loss_dict[key] = float(value)
    if self.initial_reference_diagnostics is not None:
      for key, value in self.initial_reference_diagnostics.items():
        loss_dict[f"initial_{key}"] = value
    entropy_code = getattr(self._raw_actor, "bottleneck_entropy", None)
    if entropy_code is not None:
      value = entropy_code()
      if value is not None:
        loss_dict["fsq_entropy"] = float(value)

    self.storage.clear()
    return loss_dict

  def _adapt_learning_rate(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
    acq_mask: torch.Tensor,
  ) -> float:
    """Adaptive-KL schedule using acquisition rows and a true global mean.

    Returns the global mean KL it acted on, so ``update`` can report it.
    """
    with torch.inference_mode():
      kl = self.actor.get_kl_divergence(old_params, new_params)
      selected = kl[acq_mask]
      stats = torch.stack(
        (
          selected.sum(),
          torch.tensor(float(selected.numel()), device=self.device),
        )
      )
      if self.is_multi_gpu:
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
      kl_mean = stats[0] / stats[1].clamp_min(1.0)
      if self.gpu_global_rank == 0:
        if kl_mean > self.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif self.desired_kl / 2.0 > kl_mean > 0.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
      if self.is_multi_gpu:
        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
        torch.distributed.broadcast(lr_tensor, src=0)
        self.learning_rate = lr_tensor.item()
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.learning_rate
      return float(kl_mean)

  def _normalize_masked(
    self, values: torch.Tensor, mask: torch.Tensor, distributed: bool
  ) -> torch.Tensor:
    """Normalize selected rows with global sufficient statistics; zero others."""
    flat_values = values.squeeze(-1)
    selected = flat_values[mask]
    stats = torch.stack(
      (
        torch.tensor(float(selected.numel()), device=values.device),
        selected.sum(),
        selected.square().sum(),
      )
    )
    if distributed:
      torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    count = int(stats[0].item())
    if count < 2:
      raise RuntimeError(
        "Mini-batch normalization needs two acquisition rows globally."
      )
    mean = stats[1] / count
    variance = (stats[2] - count * mean.square()).clamp_min(0.0) / (count - 1)
    normalized = torch.zeros_like(flat_values)
    normalized[mask] = (flat_values[mask] - mean) / (variance.sqrt() + 1e-8)
    return normalized.unsqueeze(-1)

  # -- construction ---------------------------------------------------------

  @staticmethod
  def construct_algorithm(  # type: ignore[override]
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> PacePPO:
    """Mirror of ``PPO.construct_algorithm`` with STAR storage and ``pi_ref``."""
    alg_class: type[PacePPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

    default_sets = ["actor", "critic"]
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    alg_cfg = cfg["algorithm"]
    acquisition_fraction = alg_cfg.get("acquisition_fraction")
    base_checkpoint = alg_cfg.get("base_checkpoint")

    # Actor cfg is consumed by the constructor, so snapshot it for pi_ref first.
    actor_cfg = copy.deepcopy(cfg["actor"])
    observation_schema = _runtime_observation_schema(
      obs, env, actor_cfg, cfg["obs_groups"]
    )
    raw_env = env.unwrapped
    command = raw_env.command_manager.get_term("motion")
    if not isinstance(command, MultiMotionCommand):
      raise TypeError("ExGRMT checkpoint schema requires MultiMotionCommand.")
    require_observation_schema = not (
      actor_cfg["command_window_offsets"] is None
      and command.cfg.command_window_offsets is None
      and command.cfg.command_window_radius == 10
      and not actor_cfg["use_command_valid_mask"]
      and not actor_cfg["use_history_valid_mask"]
      and not actor_cfg["use_intent_aux"]
      and command.critic_window_offsets.numel() == 0
      and command.reconstruction_window_offsets.numel() == 0
    )

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    # Consumed by upstream construct_algorithm; pop it here too or it leaks into
    # PPO.__init__ as an unexpected keyword.
    if alg_cfg.pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)

    storage = StarRolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
      acquisition_fraction=acquisition_fraction,
      rho_topk=alg_cfg.get("rho_topk", 0.05),
      rho_star=alg_cfg.get("rho_star", 0.25),
      use_star=alg_cfg.get("use_star", True),
    )

    alg = alg_class(
      actor,
      critic,
      storage,
      observation_schema=observation_schema,
      require_observation_schema=require_observation_schema,
      device=device,
      **alg_cfg,
      multi_gpu_cfg=cfg["multi_gpu"],
    )

    if base_checkpoint:
      loaded = torch.load(base_checkpoint, map_location=device, weights_only=False)
      validate_checkpoint_observation_schema(
        cast(dict, loaded),
        observation_schema,
        require_present=require_observation_schema,
      )
      # Warm-start pi_theta from the Stage-I base policy (Algorithm 1, line 1).
      actor.load_state_dict(loaded["actor_state_dict"])
      critic.load_state_dict(loaded["critic_state_dict"])
      if alg.consolidation_enabled:
        actor_ref: MLPModel = actor_class(
          obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
        ).to(device)
        actor_ref.load_state_dict(loaded["actor_state_dict"])
        alg.attach_reference_policy(actor_ref)
        probe_size = min(int(obs.batch_size[0]), 8)
        alg.assert_reference_policy_initialized(obs[:probe_size])
      print(f"[ex-grmt] PACE: warm-started from {base_checkpoint}")
    # A Stage-II checkpoint carries actor_ref itself and can therefore be constructed
    # without the original Stage-I file for evaluation/export. A fresh Stage-II
    # training run without either source fails at the first update with a clear error.

    alg.compile(cfg.get("torch_compile_mode"))
    return alg
