"""Tests for GMTrack-specific episode boundaries."""

from types import SimpleNamespace

import torch

from gmtrack.mdp.terminations import motion_sequence_end, nonfinite_physics_state


def test_motion_sequence_end_uses_each_environments_selected_sequence_length():
  command = SimpleNamespace(
    time_steps=torch.tensor([8, 9, 2, 3]),
    motion_ids=torch.tensor([0, 0, 1, 1]),
    lib=SimpleNamespace(clip_len=torch.tensor([10, 4])),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))

  assert motion_sequence_end(env, "motion").tolist() == [False, True, False, True]


def test_nonfinite_physics_state_checks_every_proprio_source():
  data = SimpleNamespace(
    qpos=torch.zeros(4, 36),
    qvel=torch.zeros(4, 35),
    qacc=torch.zeros(4, 35),
    qacc_warmstart=torch.zeros(4, 35),
    sensordata=torch.zeros(4, 40),
  )
  data.qpos[0, 3] = torch.nan
  data.qvel[1, 8] = torch.inf
  data.qacc_warmstart[2, 4] = -torch.inf
  data.sensordata[3, 2] = torch.nan
  env = SimpleNamespace(
    num_envs=4,
    device="cpu",
    sim=SimpleNamespace(data=data),
  )

  assert nonfinite_physics_state(env).tolist() == [True, True, True, True]


def test_nonfinite_physics_state_accepts_a_finite_world():
  data = SimpleNamespace(
    qpos=torch.zeros(2, 36),
    qvel=torch.zeros(2, 35),
    qacc=torch.zeros(2, 35),
    qacc_warmstart=torch.zeros(2, 35),
    sensordata=torch.zeros(2, 40),
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    sim=SimpleNamespace(data=data),
  )

  assert not nonfinite_physics_state(env).any()
