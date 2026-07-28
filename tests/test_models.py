"""Tests for the Extreme-RGMT policy architecture."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from ex_grmt.rsl_rl.fsq import FSQ
from ex_grmt.rsl_rl.models import (
  ACTION_HIST,
  COMMAND_WINDOW,
  PROPRIO_HIST,
  ExGRMTActor,
)

H = 10
TERM_DIMS = (3, 3, 29, 29)
PROPRIO_DIM = sum(TERM_DIMS)  # 64
ACTION_DIM = 29
TOKENS = 21
TOKEN_DIM = 9 + 29  # 38
BATCH = 8

DIST_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
}


def _obs(batch: int = BATCH) -> TensorDict:
  return TensorDict(
    {
      PROPRIO_HIST: torch.randn(batch, H * PROPRIO_DIM),
      ACTION_HIST: torch.randn(batch, H * ACTION_DIM),
      COMMAND_WINDOW: torch.randn(batch, TOKENS * TOKEN_DIM),
    },
    batch_size=[batch],
  )


def _groups() -> dict[str, list[str]]:
  return {"actor": [PROPRIO_HIST, ACTION_HIST, COMMAND_WINDOW]}


def _actor(**kw) -> ExGRMTActor:
  return ExGRMTActor(
    obs=_obs(),
    obs_groups=_groups(),
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(64, 64),  # Small trunk keeps the test fast; shape logic is unchanged.
    distribution_cfg=dict(DIST_CFG),
    history_length=H,
    proprio_term_dims=TERM_DIMS,
    **kw,
  )


def test_forward_shape():
  actor = _actor()
  out = actor(_obs())
  assert out.shape == (BATCH, ACTION_DIM)


def test_latent_is_proprio_action_and_bottleneck():
  actor = _actor()
  assert actor._get_latent_dim() == PROPRIO_DIM + ACTION_DIM + actor.token_dim
  latent = actor.get_latent(_obs())
  assert latent.shape == (BATCH, PROPRIO_DIM + ACTION_DIM + actor.token_dim)


def test_proprio_sequence_unflattens_per_term_not_per_frame():
  """mjlab flattens history *within* each term.

  The incoming row is [g(H*3) | w(H*3) | q(H*29) | qd(H*29)], each block time-major.
  A naive ``view(N, H, 64)`` would silently interleave terms and timesteps -- this
  test pins the correct layout.
  """
  actor = _actor()
  n = 2
  blocks = []
  for term_idx, d in enumerate(TERM_DIMS):
    # Value encodes (term, timestep) so a mis-split is detectable.
    block = torch.arange(H, dtype=torch.float32).repeat_interleave(d) + term_idx * 100
    blocks.append(block.unsqueeze(0).repeat(n, 1))
  flat = torch.cat(blocks, dim=-1)
  assert flat.shape == (n, H * PROPRIO_DIM)

  seq = actor._proprio_sequence(flat)
  assert seq.shape == (n, H, PROPRIO_DIM)

  offset = 0
  for term_idx, d in enumerate(TERM_DIMS):
    for t in range(H):
      expected = float(t + term_idx * 100)
      assert torch.allclose(
        seq[0, t, offset : offset + d], torch.full((d,), expected)
      ), f"term {term_idx}, step {t}"
    offset += d


def test_last_history_frame_is_current_observation():
  """``o^prop_t`` fed to the trunk must be the newest history frame."""
  actor = _actor()
  obs = _obs(1)
  seq = actor._proprio_sequence(obs[PROPRIO_HIST])
  latent = actor.get_latent(obs)
  assert torch.allclose(latent[:, :PROPRIO_DIM], seq[:, -1])


def test_gradient_reaches_every_branch():
  """Mirrors PPO's actual gradient path.

  ``forward(stochastic_output=True)`` returns a *sample*, which carries no gradient;
  PPO backpropagates through ``get_output_log_prob`` instead. Testing the sample
  directly would only prove that autograd raises.
  """
  actor = _actor()
  obs = _obs()
  actions = actor(obs, stochastic_output=True)
  actor.get_output_log_prob(actions).sum().backward()
  named = dict(actor.named_parameters())
  for key in (
    "state_enc.0.weight",
    "action_enc.0.weight",
    "command_enc.0.weight",
    "hist_pos",
    "command_pos",
    "query_proj.weight",
  ):
    assert key in named, f"missing parameter {key}"
    assert named[key].grad is not None, f"no gradient for {key}"
    assert torch.isfinite(named[key].grad).all(), f"non-finite gradient for {key}"


def test_export_survives_a_backward_pass():
  """Regression: the distribution caches graph-attached tensors after a forward pass.

  ``ExGRMTActor.as_onnx`` deep-copies the whole module (unlike upstream, which only
  copies the trunk), so it has to detach the distribution first or export silently
  fails on every checkpoint save.
  """
  actor = _actor()
  obs = _obs()
  actions = actor(obs, stochastic_output=True)
  actor.get_output_log_prob(actions).sum().backward()

  exported = actor.as_onnx()
  dummy = exported.get_dummy_inputs()
  assert len(dummy) == 3
  out = exported(*dummy)
  assert out.shape == (1, ACTION_DIM)
  assert exported.input_names == [PROPRIO_HIST, ACTION_HIST, COMMAND_WINDOW]
  # The original must still be usable for training afterwards.
  assert actor.distribution is not None
  actor(obs, stochastic_output=True)


def test_export_matches_deterministic_forward():
  actor = _actor()
  actor.eval()
  obs = _obs(1)
  exported = actor.as_onnx()
  expected = actor(obs)  # stochastic_output=False -> distribution mean
  got = exported(obs[PROPRIO_HIST], obs[ACTION_HIST], obs[COMMAND_WINDOW])
  assert torch.allclose(expected, got, atol=1e-5)


def test_rejects_wrong_observation_group_order():
  with pytest.raises(ValueError, match="Order matters"):
    ExGRMTActor(
      obs=_obs(),
      obs_groups={"actor": [ACTION_HIST, PROPRIO_HIST, COMMAND_WINDOW]},
      obs_set="actor",
      output_dim=ACTION_DIM,
      distribution_cfg=dict(DIST_CFG),
    )


def test_rejects_history_dimension_mismatch():
  obs = _obs()
  obs[PROPRIO_HIST] = torch.randn(BATCH, H * PROPRIO_DIM + 1)
  with pytest.raises(ValueError, match="proprio_term_dims"):
    ExGRMTActor(
      obs=obs,
      obs_groups=_groups(),
      obs_set="actor",
      output_dim=ACTION_DIM,
      distribution_cfg=dict(DIST_CFG),
      history_length=H,
      proprio_term_dims=TERM_DIMS,
    )


def test_rejects_empirical_normalization():
  with pytest.raises(ValueError, match="LayerNorm"):
    _actor(obs_normalization=True)


def test_unified_encoder_ablation_runs():
  actor = _actor(unified_encoder=True)
  assert actor(_obs()).shape == (BATCH, ACTION_DIM)
  assert actor.num_hist_tokens == H  # Not interleaved: one token per timestep.


def test_no_fsq_ablation_is_identity():
  actor = _actor(use_fsq=False)
  assert actor(_obs()).shape == (BATCH, ACTION_DIM)
  assert actor.bottleneck_entropy() is None


##
# FSQ
##


def test_fsq_output_lies_on_the_quantization_grid():
  fsq = FSQ(dim=64, levels=5, token_dim=32)
  out = fsq(torch.randn(256, 64) * 3.0)
  half = 5 // 2
  grid = torch.arange(-half, half + 1, dtype=out.dtype) / half
  # Every output value must coincide with a grid point.
  assert torch.isclose(out.unsqueeze(-1), grid).any(dim=-1).all()
  assert out.abs().max() <= 1.0 + 1e-6


def test_fsq_straight_through_gradient():
  fsq = FSQ(dim=8, levels=5, token_dim=8)
  z = torch.randn(16, 8, requires_grad=True)
  fsq(z).sum().backward()
  assert z.grad is not None
  assert torch.isfinite(z.grad).all()
  # Rounding contributes nothing; the gradient is that of the tanh bound alone.
  assert (z.grad.abs() > 0).any()


def test_fsq_rejects_bad_shapes():
  with pytest.raises(ValueError):
    FSQ(dim=64, levels=1)
  with pytest.raises(ValueError):
    FSQ(dim=63, levels=5, token_dim=32)


def test_fsq_entropy_is_normalized():
  fsq = FSQ(dim=16, levels=5, token_dim=16)
  h = fsq.usage_entropy(torch.randn(1024, 16) * 2.0)
  assert 0.0 <= float(h) <= 1.0 + 1e-6
  # A constant input collapses to a single code -> zero entropy.
  h0 = fsq.usage_entropy(torch.zeros(1024, 16))
  assert float(h0) == pytest.approx(0.0, abs=1e-5)
