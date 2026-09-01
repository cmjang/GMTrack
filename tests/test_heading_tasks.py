"""Task-level contract for Heading-aware task variants."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from gmtrack.envs.env_cfg import (
  CAUSAL_ACTOR_WINDOW_OFFSETS,
  CAUSAL_CRITIC_WINDOW_OFFSETS,
  CAUSAL_RECONSTRUCTION_OFFSETS,
  HISTORY_LENGTH,
  TRACKED_BODIES,
)
from gmtrack.mdp.terminations import (
  bad_motion_body_pos_xy_outside_recovery,
  bad_motion_body_pos_z_only_outside_recovery,
)

BASE_STAGE1 = "GMTrack-Stage1-Flat-Unitree-G1"
HEADING_STAGE1 = "GMTrack-Stage1-Heading-Flat-Unitree-G1"
HEADING_STAGE2 = "GMTrack-Stage2-Heading-Flat-Unitree-G1"
CAUSAL_STAGE1 = "GMTrack-Stage1-Causal-Flat-Unitree-G1"
CAUSAL_HEADING_STAGE1 = "GMTrack-Stage1-Causal-Heading-Flat-Unitree-G1"


def test_heading_tasks_are_registered_without_changing_the_paper_baseline():
  tasks = list_tasks()
  assert BASE_STAGE1 in tasks
  assert HEADING_STAGE1 in tasks
  assert HEADING_STAGE2 in tasks

  baseline_env = load_env_cfg(BASE_STAGE1)
  heading_env = load_env_cfg(HEADING_STAGE1)
  baseline_rl = load_rl_cfg(BASE_STAGE1)
  heading_rl = load_rl_cfg(HEADING_STAGE1)

  assert baseline_env.commands["motion"].heading_closed_loop is False
  assert baseline_rl.actor.command_token_dim == 38
  assert heading_env.commands["motion"].heading_closed_loop is True
  assert heading_rl.actor.command_token_dim == 44
  assert "foot_pos_xy" not in baseline_env.terminations
  assert "foot_pos_z" not in baseline_env.terminations
  assert "foot_pos_xy" not in heading_env.terminations
  assert "foot_pos_z" not in heading_env.terminations


def test_heading_task_trains_yaw_recovery_but_play_starts_aligned():
  train = load_env_cfg(HEADING_STAGE1)
  play = load_env_cfg(HEADING_STAGE1, play=True)

  assert train.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
  assert play.commands["motion"].pose_range == {}


def test_causal_heading_combines_past_only_actor_and_heading_feedback():
  tasks = list_tasks()
  baseline = load_env_cfg(BASE_STAGE1)
  train = load_env_cfg(CAUSAL_HEADING_STAGE1)
  play = load_env_cfg(CAUSAL_HEADING_STAGE1, play=True)
  runner = load_rl_cfg(CAUSAL_HEADING_STAGE1)
  motion = train.commands["motion"]

  assert CAUSAL_STAGE1 in tasks
  assert CAUSAL_HEADING_STAGE1 in tasks
  assert motion.heading_closed_loop is True
  assert motion.require_causal_window is True
  assert motion.command_window_offsets == CAUSAL_ACTOR_WINDOW_OFFSETS
  assert motion.critic_window_offsets == CAUSAL_CRITIC_WINDOW_OFFSETS
  assert motion.reconstruction_window_offsets == CAUSAL_RECONSTRUCTION_OFFSETS
  assert all(offset <= 0 for offset in motion.command_window_offsets)
  assert motion.command_window_offsets[-1] == 0
  assert train.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
  assert play.commands["motion"].pose_range == {}
  assert train.rewards == baseline.rewards
  assert set(train.terminations) == set(baseline.terminations) | {
    "foot_pos_xy",
    "foot_pos_z",
  }
  assert all(
    train.terminations[name] == term for name, term in baseline.terminations.items()
  )
  foot_xy = train.terminations["foot_pos_xy"]
  assert foot_xy.func is bad_motion_body_pos_xy_outside_recovery
  assert foot_xy.params == {
    "command_name": "motion",
    "threshold": 0.2,
    "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
  }
  foot_z = train.terminations["foot_pos_z"]
  assert foot_z.func is bad_motion_body_pos_z_only_outside_recovery
  assert foot_z.params == {
    "command_name": "motion",
    "threshold": 0.15,
    "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
  }
  assert play.terminations["foot_pos_xy"].params == foot_xy.params
  assert play.terminations["foot_pos_z"].params == foot_z.params
  assert runner.actor.command_token_dim == 44
  assert runner.actor.use_command_valid_mask is True
  assert runner.actor.use_history_valid_mask is True
  assert runner.obs_groups["actor"] == (
    "proprio_hist",
    "action_hist",
    "command_window",
    "history_valid_mask",
    "past_valid_mask",
  )
  assert runner.obs_groups["critic"] == (
    "proprio_hist",
    "action_hist",
    "history_valid_mask",
    "critic",
    "command_future_window",
    "future_valid_mask",
  )
  privileged_current_dim = (
    3
    + 3
    + 29
    + 29
    + 29
    + 44
    + 1
    + len(TRACKED_BODIES) * 3
    + len(TRACKED_BODIES) * 6
    + 3
  )
  actor_dim = 640 + 290 + len(CAUSAL_ACTOR_WINDOW_OFFSETS) * 44 + HISTORY_LENGTH + 11
  critic_dim = (
    640
    + 290
    + HISTORY_LENGTH
    + privileged_current_dim
    + len(CAUSAL_CRITIC_WINDOW_OFFSETS) * 44
    + len(CAUSAL_CRITIC_WINDOW_OFFSETS)
  )
  assert actor_dim == 1435
  assert critic_dim == 1522
  assert len(CAUSAL_RECONSTRUCTION_OFFSETS) * 44 == 132
  assert runner.experiment_name == "gmtrack_stage1_causal_heading"


def test_heading_stage2_requires_the_matching_heading_actor_shape():
  stage1 = load_rl_cfg(HEADING_STAGE1)
  stage2 = load_rl_cfg(HEADING_STAGE2)

  assert stage1.actor.command_token_dim == stage2.actor.command_token_dim == 44
  assert stage1.experiment_name == "gmtrack_stage1_heading"
  assert stage2.experiment_name == "gmtrack_stage2_heading"
