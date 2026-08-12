"""Task-level contract for the opt-in SONIC-style heading variant."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

BASE_STAGE1 = "ExGRMT-Stage1-Flat-Unitree-G1"
HEADING_STAGE1 = "ExGRMT-Stage1-Heading-Flat-Unitree-G1"
HEADING_STAGE2 = "ExGRMT-Stage2-Heading-Flat-Unitree-G1"


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


def test_heading_task_trains_yaw_recovery_but_play_starts_aligned():
  train = load_env_cfg(HEADING_STAGE1)
  play = load_env_cfg(HEADING_STAGE1, play=True)

  assert train.commands["motion"].pose_range == {"yaw": (-0.2, 0.2)}
  assert play.commands["motion"].pose_range == {}


def test_heading_stage2_requires_the_matching_heading_actor_shape():
  stage1 = load_rl_cfg(HEADING_STAGE1)
  stage2 = load_rl_cfg(HEADING_STAGE2)

  assert stage1.actor.command_token_dim == stage2.actor.command_token_dim == 44
  assert stage1.experiment_name == "ex_grmt_stage1_heading"
  assert stage2.experiment_name == "ex_grmt_stage2_heading"
