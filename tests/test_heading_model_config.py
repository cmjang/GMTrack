"""Compatibility tests for open-loop and heading command-token actors."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from gmtrack.rl_cfgs import stage1_runner_cfg, stage2_runner_cfg
from gmtrack.rsl_rl.config import GMTrackActorCfg
from gmtrack.rsl_rl.models import (
  ACTION_HIST,
  COMMAND_WINDOW,
  PROPRIO_HIST,
  GMTrackActor,
)
from gmtrack.rsl_rl.runner import _command_token_layout

HISTORY_LENGTH = 10
PROPRIO_TERM_DIMS = (3, 3, 29, 29)
ACTION_DIM = 29
COMMAND_TOKENS = 21


def _obs(command_token_dim: int) -> TensorDict:
  batch = 2
  return TensorDict(
    {
      PROPRIO_HIST: torch.randn(batch, HISTORY_LENGTH * sum(PROPRIO_TERM_DIMS)),
      ACTION_HIST: torch.randn(batch, HISTORY_LENGTH * ACTION_DIM),
      COMMAND_WINDOW: torch.randn(batch, COMMAND_TOKENS * command_token_dim),
    },
    batch_size=[batch],
  )


def _actor(command_token_dim: int, obs_token_dim: int | None = None) -> GMTrackActor:
  obs = _obs(command_token_dim if obs_token_dim is None else obs_token_dim)
  return GMTrackActor(
    obs=obs,
    obs_groups={
      "actor": [PROPRIO_HIST, ACTION_HIST, COMMAND_WINDOW],
    },
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=(32,),
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    history_length=HISTORY_LENGTH,
    proprio_term_dims=PROPRIO_TERM_DIMS,
    token_dim=32,
    num_heads=4,
    command_token_dim=command_token_dim,
  )


def test_default_actor_config_and_stage_factories_remain_38d():
  assert GMTrackActorCfg().command_token_dim == 38
  assert stage1_runner_cfg().actor.command_token_dim == 38
  assert stage1_runner_cfg().experiment_name == "gmtrack_stage1"
  assert stage2_runner_cfg().actor.command_token_dim == 38
  assert stage2_runner_cfg().experiment_name == "gmtrack_stage2"


def test_heading_factories_use_44d_and_unique_default_names():
  stage1 = stage1_runner_cfg(heading_closed_loop=True)
  stage2 = stage2_runner_cfg(heading_closed_loop=True)

  assert stage1.actor.command_token_dim == 44
  assert stage1.experiment_name == "gmtrack_stage1_heading"
  assert stage2.actor.command_token_dim == 44
  assert stage2.experiment_name == "gmtrack_stage2_heading"

  explicit = stage2_runner_cfg(
    heading_closed_loop=True, experiment_name="custom_heading_run"
  )
  assert explicit.experiment_name == "custom_heading_run"


def test_causal_heading_factory_keeps_44d_heading_tokens():
  cfg = stage1_runner_cfg(causal_online=True, heading_closed_loop=True)

  assert cfg.actor.command_token_dim == 44
  assert cfg.actor.use_command_valid_mask is True
  assert cfg.actor.use_history_valid_mask is True
  assert cfg.experiment_name == "gmtrack_stage1_causal_heading"


@pytest.mark.parametrize("command_token_dim", [38, 44])
def test_actor_and_export_derive_command_shape_from_config(command_token_dim: int):
  actor = _actor(command_token_dim)
  obs = _obs(command_token_dim)

  assert actor.command_token_dim == command_token_dim
  assert actor.num_command_tokens == COMMAND_TOKENS
  assert actor(obs).shape == (2, ACTION_DIM)

  exported = actor.as_onnx()
  assert exported.input_names == [PROPRIO_HIST, ACTION_HIST, COMMAND_WINDOW]
  assert exported.get_dummy_inputs()[2].shape == (
    1,
    COMMAND_TOKENS * command_token_dim,
  )


def test_actor_rejects_environment_with_different_command_token_width():
  with pytest.raises(ValueError, match="actor and environment use the same"):
    _actor(command_token_dim=44, obs_token_dim=38)


def test_metadata_layout_describes_open_and_heading_tokens():
  assert _command_token_layout(38, 29) == [
    "v_ref[3]",
    "w_ref[3]",
    "g_ref[3]",
    "q_ref[29]",
  ]
  assert _command_token_layout(44, 29) == [
    "v_ref[3]",
    "w_ref[3]",
    "g_ref[3]",
    "q_ref[29]",
    "root_ori_error[6]",
  ]
