"""Focused tests for PACE objective routing, distributed state, and resume policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from ex_grmt.provenance import (
  CHECKPOINT_OBSERVATION_SCHEMA_KEY,
  build_observation_schema,
)
from ex_grmt.rsl_rl.ppo_pace import PacePPO
from ex_grmt.rsl_rl.storage import TRACKING_FAILURES_EXTRA, StarBatch

TEST_OBSERVATION_SCHEMA = build_observation_schema(
  actor_observation_groups=["x"],
  critic_observation_groups=["x"],
  observation_group_widths={"x": 1},
  command_window_offsets=[0],
  critic_window_offsets=[],
  reconstruction_window_offsets=[],
  command_fps=50.0,
  command_token_dim=1,
  heading_closed_loop=False,
  history_length=1,
  proprio_term_names=["x"],
  proprio_term_dims=[1],
  action_dim=1,
  use_past_valid_mask=False,
  use_history_valid_mask=False,
  use_future_valid_mask=False,
  use_reconstruction_valid_mask=False,
  command_position_encoding="legacy_sinusoidal_slot_index",
  use_intent_aux=False,
  intent_latent_dim=64,
)


def _pace(*args, **kwargs) -> PacePPO:
  return PacePPO(
    *args,
    observation_schema=TEST_OBSERVATION_SCHEMA,
    require_observation_schema=False,
    **kwargs,
  )


def _bare_algorithm() -> PacePPO:
  alg = object.__new__(PacePPO)
  alg.device = "cpu"
  alg.is_multi_gpu = False
  alg.gpu_global_rank = 0
  alg.gpu_world_size = 1
  alg.observation_schema = TEST_OBSERVATION_SCHEMA
  alg.require_observation_schema = False
  return alg


def test_minibatch_normalization_uses_acquisition_only():
  alg = _bare_algorithm()
  values = torch.tensor([[1.0], [3.0], [1.0e9], [-1.0e9]])
  mask = torch.tensor([True, True, False, False])
  normalized = alg._normalize_masked(values, mask, distributed=False).squeeze(-1)

  assert normalized[:2].tolist() == pytest.approx([-0.7071068, 0.7071068])
  assert torch.count_nonzero(normalized[2:]) == 0


def test_intent_losses_use_the_existing_acquisition_mask_only():
  reconstruction = torch.tensor([1.0, 3.0, 1.0e6, 1.0e6])
  kl = torch.tensor([2.0, 4.0, 1.0e6, 1.0e6])
  valid_counts = torch.tensor([3, 2, 3, 3])
  acq_mask = torch.tensor([True, True, False, False])

  recon_loss, kl_loss, valid_mean = PacePPO._masked_intent_losses(
    reconstruction, kl, valid_counts, acq_mask
  )

  assert recon_loss.item() == pytest.approx(2.0)
  assert kl_loss.item() == pytest.approx(3.0)
  assert valid_mean.item() == pytest.approx(2.5)


def test_intent_reconstruction_rejects_an_all_invalid_acquisition_batch():
  with pytest.raises(RuntimeError, match="no valid future"):
    PacePPO._masked_intent_losses(
      torch.zeros(2),
      torch.zeros(2),
      torch.tensor([0, 3]),
      torch.tensor([True, False]),
    )


def test_lambda_uses_global_valid_counts(monkeypatch):
  alg = _bare_algorithm()
  alg.is_multi_gpu = True
  alg.storage = SimpleNamespace(
    valid_sample_counts=lambda: (10, 30), valid_sample_diagnostics=lambda: {}
  )
  alg.consolidation_enabled = True
  alg.rho_bar = 0.6
  alg.beta = 0.0
  alg.rho_ref = 0.6
  alg.lambda_base = 0.3
  alg.kappa = 5.0
  alg.fixed_lambda_con = None

  def fake_all_reduce(counts, op):
    del op
    # The other rank contributes (30 acquisition, 10 consolidation).
    counts += torch.tensor([30, 10])

  monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
  n_acq, n_con = alg._update_lambda_con()

  assert (n_acq, n_con) == (40, 40)
  assert alg.rho_bar == pytest.approx(0.5)
  assert alg.lambda_con == pytest.approx(0.3)


@pytest.mark.parametrize(
  ("extras", "expected", "derived"),
  (
    (
      {TRACKING_FAILURES_EXTRA: torch.tensor([True, False])},
      [True, False],
      False,
    ),
    (
      {"time_outs": torch.tensor([True, False])},
      [False, True],
      True,
    ),
  ),
)
def test_process_env_step_records_separate_failure_mask(
  monkeypatch, extras, expected, derived
):
  alg = _bare_algorithm()
  recorded = {}

  def record(mask, *, derived):
    recorded["mask"] = mask
    recorded["derived"] = derived

  alg.storage = SimpleNamespace(record_tracking_failures=record)
  monkeypatch.setattr(PPO, "process_env_step", lambda *args, **kwargs: None)
  alg.process_env_step(
    TensorDict({}, batch_size=[2]),
    torch.zeros(2),
    torch.tensor([True, True]),
    extras,
  )

  assert recorded["mask"].tolist() == expected
  assert recorded["derived"] is derived


def test_process_env_step_rejects_missing_failure_provenance(monkeypatch):
  alg = _bare_algorithm()
  alg.storage = SimpleNamespace(record_tracking_failures=lambda *args: None)
  monkeypatch.setattr(PPO, "process_env_step", lambda *args, **kwargs: None)
  with pytest.raises(RuntimeError, match="true tracking failure mask"):
    alg.process_env_step(
      TensorDict({}, batch_size=[2]),
      torch.zeros(2),
      torch.zeros(2),
      {},
    )


@pytest.mark.parametrize(
  ("lr_policy", "expected_lr"), (("checkpoint", 1.0e-5), ("config", 1.0e-3))
)
def test_resume_learning_rate_policy_is_explicit(monkeypatch, lr_policy, expected_lr):
  alg = _bare_algorithm()
  parameter = torch.nn.Parameter(torch.tensor(1.0))
  alg.optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
  alg.learning_rate = 1.0e-3
  alg.rho_bar = 0.6
  alg.lambda_con = 0.3
  alg.consolidation_enabled = False
  alg.actor_ref = None

  checkpoint_optimizer = torch.optim.Adam(
    [torch.nn.Parameter(torch.tensor(2.0))], lr=1.0e-5
  )
  loaded = {
    "optimizer_state_dict": checkpoint_optimizer.state_dict(),
    "pace_state_dict": {"version": 1, "rho_bar": 0.75, "lambda_con": 0.8},
  }

  def fake_ppo_load(self, loaded_dict, load_cfg, strict):
    del strict
    if load_cfg.get("optimizer"):
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    return bool(load_cfg.get("iteration"))

  monkeypatch.setattr(PPO, "load", fake_ppo_load)
  alg.load(
    loaded,
    {"optimizer": True, "iteration": False, "optimizer_lr": lr_policy},
    strict=True,
  )

  assert alg.learning_rate == pytest.approx(expected_lr)
  assert alg.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
  assert alg.rho_bar == pytest.approx(0.75)
  assert alg.lambda_con == pytest.approx(0.8)


def test_save_includes_pace_state(monkeypatch):
  alg = _bare_algorithm()
  alg.rho_bar = 0.72
  alg.lambda_con = 0.9
  alg.learning_rate = 2.0e-4
  alg.actor_ref = None
  monkeypatch.setattr(PPO, "save", lambda self: {"upstream": True})

  saved = alg.save()
  assert saved["upstream"] is True
  assert (
    saved[CHECKPOINT_OBSERVATION_SCHEMA_KEY] == TEST_OBSERVATION_SCHEMA
  )
  assert saved["pace_state_dict"] == {
    "version": 1,
    "rho_bar": 0.72,
    "lambda_con": 0.9,
    "learning_rate": 2.0e-4,
  }


class _TinyActor(torch.nn.Module):
  is_recurrent = False

  def __init__(self) -> None:
    super().__init__()
    self.linear = torch.nn.Linear(1, 1)
    self._mean = torch.empty(0)
    self.stochastic_calls: list[bool] = []

  def forward(self, obs, stochastic_output=False):
    self.stochastic_calls.append(stochastic_output)
    self._mean = self.linear(obs["x"])
    # Deliberately make samples differ from the raw mean so Eq. 15 tests can catch
    # accidental use of a stochastic action in the consolidation loss.
    return self._mean + 100.0 if stochastic_output else self._mean

  @property
  def output_distribution_params(self):
    return (self._mean, torch.ones_like(self._mean))

  @property
  def output_entropy(self):
    return self._mean.squeeze(-1) * 0.0 + 1.0

  @property
  def output_mean(self):
    return self._mean

  def get_output_log_prob(self, actions):
    return -0.5 * (actions - self._mean).square().sum(-1)

  def get_kl_divergence(self, old_params, new_params):
    return (old_params[0] - new_params[0]).square().sum(-1)


class _TinyCritic(torch.nn.Module):
  is_recurrent = False

  def __init__(self) -> None:
    super().__init__()
    self.linear = torch.nn.Linear(1, 1)

  def forward(self, obs):
    return self.linear(obs["x"])


class _OneBatchStorage:
  def __init__(self, batch: StarBatch) -> None:
    self.batch = batch
    self.last_star_pool_size = 0

  def valid_sample_counts(self):
    return 2, 2

  def valid_sample_diagnostics(self):
    return {}

  def star_mini_batch_generator(self, num_mini_batches, num_epochs):
    assert (num_mini_batches, num_epochs) == (1, 1)
    yield self.batch

  def clear(self):
    pass


def test_unsupported_update_extensions_fail_fast():
  recurrent_actor = _TinyActor()
  recurrent_actor.is_recurrent = True
  with pytest.raises(ValueError, match="recurrent actor/critic"):
    _pace(recurrent_actor, _TinyCritic(), None)  # type: ignore[arg-type]

  with pytest.raises(ValueError, match="RND"):
    _pace(
      _TinyActor(),
      _TinyCritic(),
      None,  # type: ignore[arg-type]
      rnd_cfg={"unsupported": True},
    )

  with pytest.raises(ValueError, match="symmetry"):
    _pace(
      _TinyActor(),
      _TinyCritic(),
      None,  # type: ignore[arg-type]
      symmetry_cfg={"unsupported": True},
    )


def test_reference_policy_is_frozen_and_initially_identical():
  actor = _TinyActor()
  critic = _TinyCritic()
  alg = _pace(actor, critic, None, desired_kl=None)  # type: ignore[arg-type]
  reference = _TinyActor()
  reference.load_state_dict(actor.state_dict())
  alg.attach_reference_policy(reference)

  diagnostics = alg.assert_reference_policy_initialized(
    TensorDict({"x": torch.tensor([[0.25], [-0.5]])}, [2])
  )
  assert diagnostics["deterministic_mean_max_abs_diff"] == pytest.approx(0.0)
  assert diagnostics["reference_kl_max"] == pytest.approx(0.0)
  assert not reference.training
  assert all(not parameter.requires_grad for parameter in reference.parameters())
  optimizer_ids = {
    id(parameter)
    for group in alg.optimizer.param_groups
    for parameter in group["params"]
  }
  assert not optimizer_ids & {id(parameter) for parameter in reference.parameters()}

  reference.linear.bias.data.add_(1.0)
  with pytest.raises(
    RuntimeError, match="requires current actor and actor_ref to match"
  ):
    alg.assert_reference_policy_initialized(
      TensorDict({"x": torch.tensor([[0.25]])}, [1])
    )


def _updated_parameters(consolidation_target: float):
  torch.manual_seed(7)
  actor = _TinyActor()
  critic = _TinyCritic()
  obs = TensorDict({"x": torch.tensor([[0.1], [0.2], [0.3], [0.4]])}, [4])
  returns = torch.tensor(
    [[1.0], [2.0], [consolidation_target], [-consolidation_target]]
  )
  advantages = torch.tensor(
    [[0.5], [-0.5], [consolidation_target], [-consolidation_target]]
  )
  batch = StarBatch(
    observations=obs,
    actions=torch.zeros(4, 1),
    values=torch.zeros(4, 1),
    advantages=advantages,
    returns=returns,
    old_actions_log_prob=torch.zeros(4),
    old_distribution_params=(torch.zeros(4, 1), torch.ones(4, 1)),
    acq_mask=torch.tensor([True, True, False, False]),
  )
  storage = _OneBatchStorage(batch)
  alg = _pace(
    actor,
    critic,
    storage,  # type: ignore[arg-type]
    acquisition_fraction=0.5,
    num_learning_epochs=1,
    num_mini_batches=1,
    learning_rate=1.0e-2,
    desired_kl=None,
    entropy_coef=0.01,
    consolidation_enabled=True,
    use_clipped_value_loss=False,
  )
  reference = _TinyActor()
  for parameter in reference.parameters():
    parameter.data.zero_()
  alg.attach_reference_policy(reference)
  losses = alg.update()
  return (
    tuple(p.detach().clone() for p in actor.parameters()),
    tuple(p.detach().clone() for p in critic.parameters()),
    losses,
  )


def test_consolidation_targets_do_not_affect_ppo_or_value_loss():
  actor_a, critic_a, losses_a = _updated_parameters(10.0)
  actor_b, critic_b, losses_b = _updated_parameters(1.0e9)

  for left, right in zip(actor_a, actor_b, strict=True):
    torch.testing.assert_close(left, right)
  for left, right in zip(critic_a, critic_b, strict=True):
    torch.testing.assert_close(left, right)
  assert losses_a["surrogate"] == pytest.approx(losses_b["surrogate"])
  assert losses_a["value"] == pytest.approx(losses_b["value"])
  assert losses_a["actor_grad_norm_mean"] > 0.0
  assert losses_a["critic_grad_norm_mean"] > 0.0
  assert losses_a["actor_grad_norm_max"] == pytest.approx(
    losses_a["actor_grad_norm_mean"]
  )
  assert losses_a["critic_grad_norm_max"] == pytest.approx(
    losses_a["critic_grad_norm_mean"]
  )
  assert losses_a["actor_grad_norm"] == losses_a["actor_grad_norm_mean"]
  assert losses_a["critic_grad_norm"] == losses_a["critic_grad_norm_mean"]


def test_eq15_uses_deterministic_policy_means_for_both_actors():
  torch.manual_seed(11)
  actor = _TinyActor()
  critic = _TinyCritic()
  obs = TensorDict({"x": torch.tensor([[0.1], [0.2], [0.3], [0.4]])}, [4])
  batch = StarBatch(
    observations=obs,
    actions=torch.zeros(4, 1),
    values=torch.zeros(4, 1),
    advantages=torch.tensor([[0.5], [-0.5], [0.0], [0.0]]),
    returns=torch.zeros(4, 1),
    old_actions_log_prob=torch.zeros(4),
    old_distribution_params=(torch.zeros(4, 1), torch.ones(4, 1)),
    acq_mask=torch.tensor([True, True, False, False]),
  )
  alg = _pace(
    actor,
    critic,
    _OneBatchStorage(batch),  # type: ignore[arg-type]
    acquisition_fraction=0.5,
    num_learning_epochs=1,
    num_mini_batches=1,
    desired_kl=None,
  )
  reference = _TinyActor()
  reference.load_state_dict(actor.state_dict())
  alg.attach_reference_policy(reference)

  losses = alg.update()

  assert actor.stochastic_calls == [True]
  assert reference.stochastic_calls == [False]
  assert losses["consolidation"] == pytest.approx(0.0)


def _adaptive_update(old_mean: float, learning_rate: float) -> dict[str, float]:
  """Run one PACE update with the adaptive-KL schedule live.

  ``_TinyActor.get_kl_divergence`` is the squared mean difference, so ``old_mean``
  sets the KL the schedule reacts to.
  """
  torch.manual_seed(7)
  actor = _TinyActor()
  critic = _TinyCritic()
  obs = TensorDict({"x": torch.tensor([[0.1], [0.2], [0.3], [0.4]])}, [4])
  batch = StarBatch(
    observations=obs,
    actions=torch.zeros(4, 1),
    values=torch.zeros(4, 1),
    advantages=torch.tensor([[0.5], [-0.5], [0.0], [0.0]]),
    returns=torch.zeros(4, 1),
    old_actions_log_prob=torch.zeros(4),
    old_distribution_params=(
      torch.full((4, 1), old_mean),
      torch.ones(4, 1),
    ),
    acq_mask=torch.tensor([True, True, False, False]),
  )
  alg = _pace(
    actor,
    critic,
    _OneBatchStorage(batch),  # type: ignore[arg-type]
    acquisition_fraction=0.5,
    num_learning_epochs=1,
    num_mini_batches=1,
    learning_rate=learning_rate,
    desired_kl=0.01,
    schedule="adaptive",
    entropy_coef=0.01,
    consolidation_enabled=False,
  )
  return alg.update()


def test_update_reports_the_kl_the_schedule_reacted_to():
  """The schedule's input must be observable, not just its effect on the rate.

  A run pinned to rsl-rl's 1e-5 floor looks identical whether the KL is marginally
  or catastrophically over target, and the two need opposite fixes.
  """
  losses = _adaptive_update(old_mean=0.0, learning_rate=1.0e-3)
  kl = losses["kl"]
  assert kl == kl and kl >= 0.0  # finite

  # The reported KL must be the one the schedule acted on, not a separate estimate:
  # the rate it produced has to follow from it under the documented rule.
  if kl > 2 * 0.01:
    expected = 1.0e-3 / 1.5
  elif kl < 0.01 / 2:
    expected = 1.0e-3 * 1.5
  else:
    expected = 1.0e-3
  assert losses["learning_rate"] == pytest.approx(expected)


def test_large_kl_cuts_the_rate_and_stops_at_the_floor():
  over = _adaptive_update(old_mean=10.0, learning_rate=1.0e-3)
  assert over["kl"] > 2 * 0.01
  assert over["learning_rate"] == pytest.approx(1.0e-3 / 1.5)

  floored = _adaptive_update(old_mean=10.0, learning_rate=1.0e-5)
  assert floored["learning_rate"] == pytest.approx(1.0e-5), "1e-5 is rsl-rl's floor"


def test_kl_is_nan_when_the_schedule_is_off():
  """Fixed-rate runs have no KL to report; the field must not silently read zero."""
  torch.manual_seed(7)
  _, _, losses = _updated_parameters(10.0)
  assert losses["kl"] != losses["kl"]
