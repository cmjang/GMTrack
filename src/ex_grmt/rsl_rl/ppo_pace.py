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

INTERPRETATION (not pinned down by the paper, recorded here and in CLAUDE.md):

* *Valid sample* = a rollout step at or before the environment's first termination.
  See :meth:`~ex_grmt.rsl_rl.storage.StarRolloutStorage.valid_sample_counts`.
* The **value** loss is computed over *both* groups -- the critic supplies GAE targets
  for consolidation environments too, so starving it there would bias their
  advantages. The **surrogate** and **entropy** terms use acquisition rows only;
  including consolidation rows in the surrogate is exactly the "Mixed Training"
  ablation the paper compares against.
* The adaptive-KL learning-rate schedule uses the whole mini-batch. Both groups were
  collected on-policy by the same ``pi_theta_old``, so their KL is equally valid, and
  restricting it would only add variance.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from ex_grmt.rsl_rl.storage import StarRolloutStorage


class PacePPO(PPO):
  """PPO with PACE consolidation and STAR mini-batch resampling."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: StarRolloutStorage,
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
    **kwargs,
  ) -> None:
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

    # Eq. (18): the smoothed ratio starts at rho_ref so lambda_con starts at lambda_base.
    self.rho_bar = rho_ref
    self.lambda_con = lambda_base if fixed_lambda_con is None else fixed_lambda_con

    self.actor_ref: MLPModel | None = None

  # -- reference policy -----------------------------------------------------

  def attach_reference_policy(self, actor_ref: MLPModel) -> None:
    """Install the frozen Stage-I policy used by the consolidation loss.

    Kept out of ``self.optimizer`` (built in ``PPO.__init__`` from actor+critic only)
    and out of ``save()`` (which enumerates actor/critic/optimizer explicitly), so
    checkpoints stay the same size. Resuming re-creates it from ``base_checkpoint``.
    """
    actor_ref.eval()
    for p in actor_ref.parameters():
      p.requires_grad_(False)
    self.actor_ref = actor_ref

  def train_mode(self) -> None:
    super().train_mode()
    if self.actor_ref is not None:
      self.actor_ref.eval()  # Never leaves eval.

  # -- returns --------------------------------------------------------------

  def compute_returns(self, obs: TensorDict) -> None:
    """GAE, then STAR's difficulty-conditioned advantage normalization (Eq. 21-23)."""
    st = self.storage
    critic_hidden_state = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    self.critic.reset(hidden_state=critic_hidden_state)

    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = (
        last_values
        if step == st.num_transitions_per_env - 1
        else st.values[step + 1]
      )
      next_is_not_terminal = 1.0 - st.dones[step].float()
      delta = (
        st.rewards[step] + next_is_not_terminal * self.gamma * next_values
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
      st.normalize_advantages_by_difficulty()
    else:
      st.advantages = (st.advantages - st.advantages.mean()) / (
        st.advantages.std() + 1e-8
      )

  # -- consolidation weight -------------------------------------------------

  def _update_lambda_con(self) -> tuple[int, int]:
    """Eq. (17)-(19). Returns the raw valid-sample counts for logging."""
    n_acq, n_con = self.storage.valid_sample_counts()
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

  def update(self) -> dict[str, float]:  # noqa: C901 - mirrors upstream structure
    n_acq, n_con = self._update_lambda_con()

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_consolidation = 0.0

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
          adv = batch.advantages
          batch.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        self._adapt_learning_rate(batch.old_distribution_params, distribution_params)

      # -- acquisition: clipped PPO surrogate over E_A rows only (Eq. 16) --
      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      adv = torch.squeeze(batch.advantages)
      surrogate = -adv * ratio
      surrogate_clipped = -adv * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped)[acq_mask].mean()

      # -- value loss over both groups (see module docstring) --
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_loss = torch.max(
          (values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)
        ).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy[acq_mask].mean()
      )

      # -- consolidation: BC against the frozen Stage-I policy (Eq. 15) --
      if has_con:
        assert self.actor_ref is not None, (
          "PACE consolidation is enabled but no reference policy is attached; "
          "pass `base_checkpoint` so construct_algorithm can build pi_ref."
        )
        con_obs = batch.observations[con_mask]
        with torch.no_grad():
          ref_actions = self.actor_ref(con_obs)
        cur_actions = self.actor.output_mean[con_mask]
        consolidation_loss = (cur_actions - ref_actions).pow(2).sum(-1).mean()
        loss = loss + self.lambda_con * consolidation_loss
        mean_consolidation += consolidation_loss.item()

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()

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
    }
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
  ) -> None:
    """Upstream's adaptive-KL schedule, extracted so ``update`` stays readable."""
    with torch.inference_mode():
      kl_mean = torch.mean(self.actor.get_kl_divergence(old_params, new_params))
      if self.is_multi_gpu:
        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
        kl_mean /= self.gpu_world_size
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
      actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"]
    )

    if base_checkpoint:
      loaded = torch.load(base_checkpoint, map_location=device, weights_only=False)
      # Warm-start pi_theta from the Stage-I base policy (Algorithm 1, line 1).
      actor.load_state_dict(loaded["actor_state_dict"])
      critic.load_state_dict(loaded["critic_state_dict"])
      if alg.consolidation_enabled:
        actor_ref: MLPModel = actor_class(
          obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
        ).to(device)
        actor_ref.load_state_dict(loaded["actor_state_dict"])
        alg.attach_reference_policy(actor_ref)
      print(f"[ex-grmt] PACE: warm-started from {base_checkpoint}")
    elif alg.consolidation_enabled:
      raise ValueError(
        "PACE consolidation needs `algorithm.base_checkpoint` pointing at a Stage-I "
        "checkpoint (it supplies both the pi_theta warm start and pi_ref)."
      )

    alg.compile(cfg.get("torch_compile_mode"))
    return alg
