"""Contracts for the causal online-teleoperation tasks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from gmtrack.envs.env_cfg import (
  CAUSAL_ACTOR_WINDOW_OFFSETS,
  CAUSAL_CRITIC_WINDOW_OFFSETS,
  CAUSAL_RECONSTRUCTION_OFFSETS,
  COMMAND_WINDOW_NOISE,
  COMMAND_WINDOW_RADIUS,
  HISTORY_LENGTH,
  TRACKED_BODIES,
  command_window_noise,
  make_gmtrack_env_cfg,
)
from gmtrack.mdp.commands import (
  MultiMotionCommandCfg,
  _parse_critic_window_offsets,
  _parse_reconstruction_window_offsets,
  _parse_window_offsets,
)
from gmtrack.mdp.motion_library import MotionLibrary
from gmtrack.mdp.observations import (
  executed_history_valid_mask,
  motion_command_future_valid_mask,
  motion_command_past_valid_mask,
  motion_command_token,
)
from gmtrack.mdp.terminations import (
  bad_motion_body_pos_xy_outside_recovery,
  bad_motion_body_pos_z_only_outside_recovery,
)
from gmtrack.rl_cfgs import INTENT_KL_COEF, INTENT_RECONSTRUCTION_COEF

BASE_STAGE1 = "GMTrack-Stage1-Flat-Unitree-G1"
CAUSAL_STAGE1 = "GMTrack-Stage1-Causal-Flat-Unitree-G1"
CAUSAL_RECOVERY_STAGE1 = "GMTrack-Stage1-Causal-Recovery-Flat-Unitree-G1"
CAUSAL_NOMASK_STAGE1 = "GMTrack-Stage1-CausalNoMask-Flat-Unitree-G1"
CAUSAL_STAGE2 = "GMTrack-Stage2-Causal-Flat-Unitree-G1"
TOKEN_DIM = 38
REGISTERED_CAUSAL_TOKEN_DIM = 44
CAUSAL_ACTOR_OFFSETS = (-32, -24, -16, -12, -8, -6, -4, -3, -2, -1, 0)
CAUSAL_CRITIC_OFFSETS = (1, 2, 4, 8, 16, 32, 64)
CAUSAL_RECONSTRUCTION_TARGETS = (5, 10, 20)

LEGACY_TASKS = (
  "GMTrack-Stage1-Flat-Unitree-G1",
  "GMTrack-Stage1-Recovery-Flat-Unitree-G1",
  "GMTrack-Stage1-Heading-Flat-Unitree-G1",
  "GMTrack-Stage1-Heading-Recovery-Flat-Unitree-G1",
  "GMTrack-Stage2-Flat-Unitree-G1",
  "GMTrack-Stage2-Heading-Flat-Unitree-G1",
  "GMTrack-Stage2-NoStar-Flat-Unitree-G1",
  "GMTrack-Stage2-NoCon-Flat-Unitree-G1",
  "GMTrack-Stage2-FixedLambda-Flat-Unitree-G1",
  "GMTrack-Stage2-UnifiedEnc-Flat-Unitree-G1",
  "GMTrack-Stage2-NoFSQ-Flat-Unitree-G1",
  "GMTrack-Finetune-Flat-Unitree-G1",
  "GMTrack-MixedTraining-Flat-Unitree-G1",
)


def _command_cfg(**overrides) -> MultiMotionCommandCfg:
  values = {
    "resampling_time_range": (1.0, 1.0),
    "manifest": "unused.json",
    "anchor_body_name": "torso_link",
    "body_names": ("pelvis",),
    "entity_name": "robot",
  }
  values.update(overrides)
  return MultiMotionCommandCfg(**values)


def _critic_dim(token_dim: int) -> int:
  return (
    3
    + 3
    + 29
    + 29
    + 29
    + token_dim
    + 1
    + len(TRACKED_BODIES) * 3
    + len(TRACKED_BODIES) * 6
    + 3
  )


def test_causal_actor_uses_exact_sparse_past_only_layout():
  cfg = make_gmtrack_env_cfg(manifest="unused.json", causal_online=True)
  motion = cfg.commands["motion"]

  assert CAUSAL_ACTOR_WINDOW_OFFSETS == CAUSAL_ACTOR_OFFSETS
  assert motion.command_window_offsets == CAUSAL_ACTOR_OFFSETS
  assert len(motion.command_window_offsets) == 11
  assert motion.command_window_offsets[-1] == 0
  assert all(offset <= 0 for offset in motion.command_window_offsets)
  assert not any(offset > 0 for offset in motion.command_window_offsets)
  assert tuple(cfg.observations["history_valid_mask"].terms) == ("mask",)


def test_causal_critic_uses_exact_strictly_future_layout():
  cfg = make_gmtrack_env_cfg(manifest="unused.json", causal_online=True)
  motion = cfg.commands["motion"]

  assert CAUSAL_CRITIC_WINDOW_OFFSETS == CAUSAL_CRITIC_OFFSETS
  assert motion.critic_window_offsets == CAUSAL_CRITIC_OFFSETS
  assert all(offset > 0 for offset in motion.critic_window_offsets)
  assert tuple(cfg.observations["command_future_window"].terms) == ("window",)
  assert tuple(cfg.observations["future_valid_mask"].terms) == ("mask",)


def test_causal_intent_targets_and_public_loss_weights_are_enabled():
  env_cfg = make_gmtrack_env_cfg(manifest="unused.json", causal_online=True)
  motion = env_cfg.commands["motion"]
  runner_cfg = load_rl_cfg(CAUSAL_STAGE1)

  assert CAUSAL_RECONSTRUCTION_OFFSETS == CAUSAL_RECONSTRUCTION_TARGETS
  assert motion.reconstruction_window_offsets == CAUSAL_RECONSTRUCTION_TARGETS
  assert tuple(env_cfg.observations["future_reconstruction_target"].terms) == (
    "target",
  )
  assert tuple(env_cfg.observations["future_reconstruction_valid_mask"].terms) == (
    "mask",
  )
  assert runner_cfg.actor.use_intent_aux is True
  assert runner_cfg.actor.intent_latent_dim == 64
  assert runner_cfg.actor.future_reconstruction_offsets == CAUSAL_RECONSTRUCTION_TARGETS
  assert (
    runner_cfg.algorithm.intent_reconstruction_coef == INTENT_RECONSTRUCTION_COEF == 0.5
  )
  assert runner_cfg.algorithm.intent_kl_coef == INTENT_KL_COEF == 0.0005


def test_command_noise_width_tracks_configured_actor_window():
  cfg = make_gmtrack_env_cfg(manifest="unused.json", causal_online=True)
  motion = cfg.commands["motion"]
  noise = cfg.observations["command_window"].terms["window"].params["magnitude"]

  assert len(noise) == len(motion.command_window_offsets) * TOKEN_DIM
  assert noise == command_window_noise(
    num_window_tokens=len(motion.command_window_offsets)
  )
  assert COMMAND_WINDOW_NOISE == command_window_noise()


@pytest.mark.parametrize(
  ("offsets", "exception", "message"),
  (
    ((-2, -1), ValueError, "contain offset 0"),
    ((-1, 0, 0), ValueError, "strictly increasing"),
    ((-1.5, 0), TypeError, "only integer offsets"),
  ),
)
def test_invalid_explicit_actor_offsets_fail(offsets, exception, message):
  cfg = _command_cfg(command_window_offsets=offsets)
  with pytest.raises(exception, match=message):
    _parse_window_offsets(cfg)


@pytest.mark.parametrize(
  ("offsets", "exception", "message"),
  (
    ((), ValueError, "critic_window_offsets"),
    ((0, 1), ValueError, "positive offsets"),
    ((1, 1), ValueError, "strictly increasing"),
    ((1.5,), TypeError, "only integer offsets"),
  ),
)
def test_invalid_critic_offsets_fail(offsets, exception, message):
  cfg = _command_cfg(critic_window_offsets=offsets)
  with pytest.raises(exception, match=message):
    _parse_critic_window_offsets(cfg)


def test_absent_critic_offsets_preserve_the_legacy_empty_window():
  assert _parse_critic_window_offsets(_command_cfg()) == ()


@pytest.mark.parametrize(
  "offsets",
  ((), (0, 5), (5, 5), (-1, 5)),
)
def test_invalid_reconstruction_offsets_fail(offsets):
  cfg = _command_cfg(reconstruction_window_offsets=offsets)
  with pytest.raises((TypeError, ValueError), match="reconstruction_window_offsets"):
    _parse_reconstruction_window_offsets(cfg)


def test_causal_actor_contract_rejects_future_offsets():
  cfg = _command_cfg(
    command_window_offsets=(-1, 0, 1),
    require_causal_window=True,
  )
  with pytest.raises(ValueError, match="forbids future actor references"):
    _parse_window_offsets(cfg)


def test_motion_command_token_uses_the_actual_zero_offset_position():
  offsets = torch.tensor([-32, -24, -16, -12, -8, -6, -4, -3, -2, -1, 0])
  tokens = torch.arange(len(offsets), dtype=torch.float32).view(1, -1, 1)
  command = SimpleNamespace(
    window_offsets=offsets,
    command_window=lambda: tokens,
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))

  actual = motion_command_token(env, "motion")
  torch.testing.assert_close(actual, torch.tensor([[10.0]]))


def test_motion_command_token_rejects_a_window_without_current_frame():
  command = SimpleNamespace(
    window_offsets=torch.tensor([-2, -1]),
    command_window=lambda: torch.zeros(1, 2, 1),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))

  with pytest.raises(ValueError, match="exactly one offset 0"):
    motion_command_token(env, "motion")


def test_observation_mask_terms_forward_command_masks_without_coercion():
  past_mask = torch.tensor([[False, True, True]])
  future_mask = torch.tensor([[True, True, False]])
  command = SimpleNamespace(
    command_window_valid_mask=lambda: past_mask,
    critic_command_window_valid_mask=lambda: future_mask,
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))

  assert motion_command_past_valid_mask(env, "motion") is past_mask
  assert motion_command_future_valid_mask(env, "motion") is future_mask
  assert past_mask.dtype is torch.bool
  assert future_mask.dtype is torch.bool


def test_executed_history_valid_mask_marks_only_post_reset_samples():
  env = SimpleNamespace(episode_length_buf=torch.tensor([0, 1, 2, 3, 12]))

  mask = executed_history_valid_mask(env, history_length=4)

  assert mask.dtype is torch.bool
  assert mask.tolist() == [
    [False, False, False, True],
    [False, False, True, True],
    [False, True, True, True],
    [True, True, True, True],
    [True, True, True, True],
  ]


def test_executed_history_valid_mask_rejects_nonpositive_history():
  env = SimpleNamespace(episode_length_buf=torch.tensor([0]))
  with pytest.raises(ValueError, match="history_length must be positive"):
    executed_history_valid_mask(env, history_length=0)


def test_valid_masks_use_physical_parent_not_logical_fragment_boundaries():
  library = object.__new__(MotionLibrary)
  library.clip_start = torch.tensor([10])
  library.clip_len = torch.tensor([20])
  library.clip_read_start = torch.tensor([0])
  library.clip_read_len = torch.tensor([50])
  motion_ids = torch.tensor([0])

  past_idx, past_valid = library.window_index_and_mask(
    motion_ids,
    torch.tensor([0]),
    torch.tensor([-12, -10, -1, 0]),
  )
  assert past_idx.tolist() == [[0, 0, 9, 10]]
  assert past_valid.tolist() == [[False, True, True, True]]

  future_idx, future_valid = library.window_index_and_mask(
    motion_ids,
    torch.tensor([19]),
    torch.tensor([1, 20, 30, 31]),
  )
  assert future_idx.tolist() == [[30, 49, 49, 49]]
  assert future_valid.tolist() == [[True, True, False, False]]


def test_existing_default_task_observation_dimensions_remain_unchanged():
  """Pin the pre-change G1 actor groups and privileged critic width."""
  cfg = make_gmtrack_env_cfg(manifest="unused.json")
  motion = cfg.commands["motion"]
  assert motion.command_window_offsets is None
  assert motion.critic_window_offsets is None
  assert motion.command_window_radius == COMMAND_WINDOW_RADIUS == 10
  assert tuple(cfg.observations) == (
    "proprio_hist",
    "action_hist",
    "command_window",
    "critic",
    "star",
  )

  actor_group_dims = {
    "proprio_hist": HISTORY_LENGTH * (3 + 3 + 29 + 29),
    "action_hist": HISTORY_LENGTH * 29,
    "command_window": len(
      cfg.observations["command_window"].terms["window"].params["magnitude"]
    ),
  }
  assert actor_group_dims == {
    "proprio_hist": 640,
    "action_hist": 290,
    "command_window": 798,
  }
  assert sum(actor_group_dims.values()) == 1728

  assert tuple(cfg.observations["critic"].terms) == (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "command_token",
    "ref_root_height",
    "body_pos",
    "body_ori",
    "base_lin_vel",
  )
  assert _critic_dim(TOKEN_DIM) == 261


@pytest.mark.parametrize("task_id", LEGACY_TASKS)
def test_every_registered_legacy_task_keeps_observation_contract(task_id):
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  heading = "Heading" in task_id
  token_dim = 44 if heading else 38

  assert env_cfg.commands["motion"].command_window_offsets is None
  assert env_cfg.commands["motion"].critic_window_offsets is None
  assert tuple(env_cfg.observations) == (
    "proprio_hist",
    "action_hist",
    "command_window",
    "critic",
    "star",
  )
  assert (
    len(env_cfg.observations["command_window"].terms["window"].params["magnitude"])
    == 21 * token_dim
  )
  assert _critic_dim(token_dim) == (267 if heading else 261)
  assert rl_cfg.obs_groups["actor"] == (
    "proprio_hist",
    "action_hist",
    "command_window",
  )
  assert rl_cfg.obs_groups["critic"] == ("critic",)
  assert rl_cfg.actor.use_command_valid_mask is False
  assert rl_cfg.actor.use_history_valid_mask is False


def test_causal_tasks_are_registered_with_asymmetric_critic_inputs():
  tasks = list_tasks()
  assert BASE_STAGE1 in tasks
  assert CAUSAL_STAGE1 in tasks
  assert CAUSAL_STAGE2 not in tasks
  assert "GMTrack-Stage1-Causal-Heading-Flat-Unitree-G1" not in tasks
  assert "GMTrack-Stage1-Causal-Heading-Recovery-Flat-Unitree-G1" not in tasks

  baseline_env = load_env_cfg(BASE_STAGE1)
  causal_env = load_env_cfg(CAUSAL_STAGE1)
  baseline_rl = load_rl_cfg(BASE_STAGE1)
  causal_rl = load_rl_cfg(CAUSAL_STAGE1)

  assert baseline_env.commands["motion"].command_window_offsets is None
  assert causal_env.commands["motion"].command_window_offsets == CAUSAL_ACTOR_OFFSETS
  assert causal_env.commands["motion"].critic_window_offsets == CAUSAL_CRITIC_OFFSETS
  assert causal_env.commands["motion"].heading_closed_loop is True
  assert causal_env.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
  assert baseline_rl.actor.use_command_valid_mask is False
  assert causal_rl.actor.command_token_dim == REGISTERED_CAUSAL_TOKEN_DIM
  assert baseline_rl.obs_groups["actor"] == (
    "proprio_hist",
    "action_hist",
    "command_window",
  )
  assert causal_rl.actor.use_command_valid_mask is True
  assert causal_rl.actor.use_history_valid_mask is True
  assert causal_rl.obs_groups["actor"] == (
    "proprio_hist",
    "action_hist",
    "command_window",
    "history_valid_mask",
    "past_valid_mask",
  )
  assert causal_rl.obs_groups["critic"] == (
    "proprio_hist",
    "action_hist",
    "history_valid_mask",
    "critic",
    "command_future_window",
    "future_valid_mask",
  )
  causal_actor_dim = (
    640
    + 290
    + len(CAUSAL_ACTOR_OFFSETS) * REGISTERED_CAUSAL_TOKEN_DIM
    + HISTORY_LENGTH
    + 11
  )
  causal_critic_dim = (
    640
    + 290
    + HISTORY_LENGTH
    + _critic_dim(REGISTERED_CAUSAL_TOKEN_DIM)
    + len(CAUSAL_CRITIC_OFFSETS) * REGISTERED_CAUSAL_TOKEN_DIM
    + len(CAUSAL_CRITIC_OFFSETS)
  )
  assert causal_actor_dim == 1435
  assert causal_critic_dim == 1522
  assert causal_rl.experiment_name == "gmtrack_stage1_causal_heading"


def test_every_registered_causal_name_defaults_to_heading():
  for task_id in (CAUSAL_STAGE1, CAUSAL_RECOVERY_STAGE1, CAUSAL_NOMASK_STAGE1):
    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)

    assert env_cfg.commands["motion"].heading_closed_loop is True
    assert env_cfg.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
    assert rl_cfg.actor.command_token_dim == REGISTERED_CAUSAL_TOKEN_DIM


def test_registered_stage1_causal_adds_only_the_sonic_foot_terminations():
  baseline = load_env_cfg(BASE_STAGE1)
  causal = load_env_cfg(CAUSAL_STAGE1)

  assert causal.rewards == baseline.rewards
  assert set(causal.terminations) == set(baseline.terminations) | {
    "foot_pos_xy",
    "foot_pos_z",
  }
  assert all(
    causal.terminations[name] == term for name, term in baseline.terminations.items()
  )
  foot_xy = causal.terminations["foot_pos_xy"]
  assert foot_xy.func is bad_motion_body_pos_xy_outside_recovery
  assert foot_xy.params["threshold"] == 0.2
  foot_z = causal.terminations["foot_pos_z"]
  assert foot_z.func is bad_motion_body_pos_z_only_outside_recovery
  assert foot_z.params["threshold"] == 0.15

  # Causal Stage II stays unavailable until a matching causal + Heading split exists.
  stage2 = load_env_cfg("GMTrack-Stage2-Flat-Unitree-G1")
  assert "foot_pos_xy" not in stage2.terminations
  assert "foot_pos_z" not in stage2.terminations


def test_causal_recovery_task_constructs_the_full_recovery_protocol():
  train = load_env_cfg(CAUSAL_RECOVERY_STAGE1)
  play = load_env_cfg(CAUSAL_RECOVERY_STAGE1, play=True)
  runner = load_rl_cfg(CAUSAL_RECOVERY_STAGE1)

  assert CAUSAL_RECOVERY_STAGE1 in list_tasks()
  assert train.commands["motion"].require_causal_window is True
  assert train.commands["motion"].heading_closed_loop is True
  assert train.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
  assert train.commands["motion"].recovery_probability == 0.15
  assert "recovery_assist" in train.events
  assert train.events["recovery_assist"].mode == "step"
  assert {"foot_pos_xy", "foot_pos_z"} <= set(train.terminations)
  assert play.commands["motion"].recovery_probability == 0.0
  assert play.commands["motion"].pose_range == {}
  assert "recovery_assist" in play.events
  assert runner.actor.command_token_dim == REGISTERED_CAUSAL_TOKEN_DIM
  assert runner.experiment_name == "gmtrack_stage1_causal_heading_recovery"
