"""Small invariants for the shared per-clip evaluation harness."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from gmtrack.scripts.rollout_eval import (
  _cartesian_error_metrics,
  _transition_horizons,
  rollout_clips,
)


def test_n_frames_have_n_minus_one_transitions():
  assert _transition_horizons(torch.tensor([2, 500, 17])).tolist() == [1, 499, 16]


def test_single_frame_clip_cannot_be_evaluated():
  with pytest.raises(ValueError, match="at least two"):
    _transition_horizons(torch.tensor([10, 1]))


def test_mpjpe_is_per_body_mean_norm_in_mm():
  # Two bodies with errors of 3 cm and 4 cm -> mean 35 mm.
  err = torch.tensor([[[0.03, 0.0, 0.0], [0.0, 0.04, 0.0]]])
  zeros = torch.zeros_like(err)
  mpjpe, _, _, _ = _cartesian_error_metrics(err, zeros, zeros)
  assert torch.allclose(mpjpe, torch.tensor([35.0]))


def test_constant_error_offset_has_zero_velocity_error():
  # A robot tracking with a constant positional bias moves at exactly the
  # reference velocity, so d_vel must vanish -- this distinguishes the error
  # *difference* from the error itself.
  err = torch.full((1, 3, 3), 0.05)
  vel = torch.zeros_like(err)
  _, d_vel, _, vel_out = _cartesian_error_metrics(err, err.clone(), vel)
  assert torch.allclose(d_vel, torch.tensor([0.0]))
  assert torch.allclose(vel_out, torch.zeros_like(err))


def test_linearly_growing_error_has_constant_velocity_and_zero_acceleration():
  # err_t = t * [1 mm, 0, 0] per body: d_vel is 1 mm/frame every step and, once the
  # first difference is in steady state, d_acc is zero.
  step_vec = torch.tensor([0.001, 0.0, 0.0]).expand(1, 2, 3)
  prev_err = 4 * step_vec
  err = 5 * step_vec
  prev_vel = step_vec.clone()
  mpjpe, d_vel, d_acc, vel_out = _cartesian_error_metrics(err, prev_err, prev_vel)
  assert torch.allclose(mpjpe, torch.tensor([5.0]))
  assert torch.allclose(d_vel, torch.tensor([1.0]))
  assert torch.allclose(d_acc, torch.tensor([0.0]), atol=1e-5)
  assert torch.allclose(vel_out, step_vec)


def test_acceleration_error_is_second_difference():
  # Velocity error jumps from 1 mm/frame to 3 mm/frame -> d_acc = 2 mm/frame^2.
  unit = torch.tensor([0.001, 0.0, 0.0]).expand(1, 1, 3)
  _, d_vel, d_acc, _ = _cartesian_error_metrics(3 * unit, 0 * unit, 1 * unit)
  assert torch.allclose(d_vel, torch.tensor([3.0]))
  assert torch.allclose(d_acc, torch.tensor([2.0]))


class _FakeSim:
  def forward(self):
    pass

  def sense(self):
    pass


class _FakeObservationManager:
  def __init__(self, command):
    self.command = command
    self._obs_buffer = {
      "target": torch.tensor([[-999.0]]),
      "history": torch.full((1, 10), -999.0),
    }
    self.history = None
    self.reset_ids = None

  def reset(self, env_ids):
    self.reset_ids = env_ids.clone()
    self._obs_buffer = None
    self.history = None

  def compute(self, update_history=False):
    if not update_history and self._obs_buffer is not None:
      return self._obs_buffer
    current_robot_frame = self.command.robot_body_pos_w[:, 1, 0]
    if update_history:
      if self.history is None:
        self.history = current_robot_frame[:, None].repeat(1, 10)
      else:
        self.history = torch.cat(
          [self.history[:, 1:], current_robot_frame[:, None]], dim=1
        )
    assert self.history is not None
    self._obs_buffer = {
      "target": self.command.time_steps[:, None].float().clone(),
      "history": self.history.clone(),
    }
    return self._obs_buffer


class _FakeCommand:
  def __init__(self, reference_child_x_by_frame=None):
    self.device = torch.device("cpu")
    self.lib = SimpleNamespace(
      clip_len=torch.tensor([3]), clips=[SimpleNamespace(name="three_frames")]
    )
    self.motion_ids = torch.zeros(1, dtype=torch.long)
    self.time_steps = torch.zeros(1, dtype=torch.long)
    self.reference_child_x_by_frame = reference_child_x_by_frame or {}
    self._robot_body_pos_w = self.body_pos_w.clone()

  @property
  def body_pos_w(self):
    frame = self.time_steps.float()
    root = torch.stack([torch.zeros_like(frame), frame * 0, 0.5 + frame * 0], dim=-1)
    child = root + torch.stack([frame, torch.zeros_like(frame), frame * 0], dim=-1)
    for injected_frame, value in self.reference_child_x_by_frame.items():
      child[frame.long() == injected_frame, 0] = value
    return torch.stack([root, child], dim=1)

  @property
  def robot_body_pos_w(self):
    return self._robot_body_pos_w

  def set_clip(self, env_ids, motion_ids, frames=0):
    self.motion_ids[env_ids] = motion_ids
    self.time_steps[env_ids] = frames
    self._robot_body_pos_w[env_ids] = self.body_pos_w[env_ids]

  def update_relative_body_poses(self):
    pass

  def _update_command(self):
    last = self.lib.clip_len[self.motion_ids] - 1
    self.time_steps.copy_(torch.minimum(self.time_steps + 1, last))


class _FakeRolloutEnv:
  def __init__(self, command, robot_child_x_by_step=None):
    self.command = command
    self.robot_child_x_by_step = robot_child_x_by_step or {}
    self.num_envs = 1
    self.unwrapped = self
    self.sim = _FakeSim()
    self.observation_manager = _FakeObservationManager(command)
    self.obs_buf = None
    self.steps = 0

  def reset(self):
    # Deliberately install a stale cache, like the real reset before set_clip.
    self.observation_manager._obs_buffer = {
      "target": torch.tensor([[-999.0]]),
      "history": torch.full((1, 10), -999.0),
    }

  def get_observations(self):
    return self.observation_manager.compute()

  def step(self, actions):
    # A perfect plant reaches the reference frame selected for this action.
    assert not actions.requires_grad
    expected = self.command.time_steps
    assert torch.equal(actions[:, 0].long(), expected) or torch.equal(
      actions, torch.zeros_like(actions)
    )
    self.command._robot_body_pos_w = self.command.body_pos_w.clone()
    if self.steps in self.robot_child_x_by_step:
      self.command._robot_body_pos_w[:, 1, 0] = self.robot_child_x_by_step[self.steps]
    self.command._update_command()
    self.observation_manager._obs_buffer = None
    obs = self.observation_manager.compute(update_history=True)
    self.steps += 1
    return obs, torch.zeros(1), torch.zeros(1, dtype=torch.long), {}


def test_rollout_uses_frame_zero_only_for_history_and_scores_final_frame():
  command = _FakeCommand()
  env = _FakeRolloutEnv(command)
  policy_inputs = []
  policy_gain = torch.nn.Parameter(torch.ones(1))

  def policy(obs):
    policy_inputs.append({key: value.clone() for key, value in obs.items()})
    # Mirrors rsl-rl's inference policy: an nn.Module whose output would carry an
    # autograd graph unless the rollout harness explicitly disables gradients.
    return obs["target"] * policy_gain

  results = rollout_clips(
    env,
    policy,
    command,
    torch.tensor([0]),
    rollouts_per_clip=1,
  )

  assert env.steps == 2  # Three frames contain exactly two transitions.
  assert [int(obs["target"].item()) for obs in policy_inputs] == [1, 2]
  # The first policy input has a frame-0 history backfilled into every slot, with no
  # value from reset's deliberately stale -999 cache.
  assert torch.equal(policy_inputs[0]["history"], torch.zeros(1, 10))
  assert command.time_steps.tolist() == [2]  # Terminal command clamps, never wraps.
  assert env.observation_manager.reset_ids.tolist() == [0]
  assert results[0].trials == 1
  assert results[0].successes == 1
  assert results[0].summary()["mpjpe_mm"] == pytest.approx(0.0)


def test_trials_alias_is_deprecated_but_compatible():
  command = _FakeCommand()
  env = _FakeRolloutEnv(command)
  with pytest.warns(DeprecationWarning, match="rollouts_per_clip"):
    results = rollout_clips(
      env, lambda obs: obs["target"], command, torch.tensor([0]), trials=1
    )
  assert results[0].trials == 1


@pytest.mark.parametrize(
  ("failure_source", "nonfinite_value"),
  [("reference", float("nan")), ("robot", float("inf"))],
)
def test_nonfinite_reference_or_robot_state_is_an_integrity_failure(
  failure_source, nonfinite_value
):
  if failure_source == "reference":
    command = _FakeCommand(reference_child_x_by_frame={1: nonfinite_value})
    env = _FakeRolloutEnv(command)
  else:
    command = _FakeCommand()
    env = _FakeRolloutEnv(command, robot_child_x_by_step={0: nonfinite_value})

  result = rollout_clips(env, lambda obs: obs["target"], command, torch.tensor([0]))[0]
  summary = result.summary()

  assert result.trials == 1
  assert result.successes == 0
  assert result.nonfinite_failures == 1
  assert summary["nonfinite_failures"] == 1
  assert summary["finite_metric_steps"] == len(result.mpjpe_mm)
  assert all(math.isfinite(float(value)) for value in summary.values())
  json.dumps(summary, allow_nan=False)


def test_nonfinite_metric_is_a_failure_even_when_input_states_are_finite():
  command = _FakeCommand()
  # float32's vector norm overflows even though this injected robot position is
  # itself finite. The derived metric must be checked independently from state.
  env = _FakeRolloutEnv(command, robot_child_x_by_step={0: -1.0e30})

  result = rollout_clips(env, lambda obs: obs["target"], command, torch.tensor([0]))[0]

  assert result.successes == 0
  assert result.nonfinite_failures == 1
  assert result.mpjpe_mm == []
  assert result.summary()["mpjpe_mm"] == 0.0
  json.dumps(result.summary(), allow_nan=False)


def test_finite_prefix_is_retained_before_nonfinite_failure():
  command = _FakeCommand()
  env = _FakeRolloutEnv(
    command,
    robot_child_x_by_step={
      0: 0.99,  # target child x is 1.0: 10 mm error over one of two bodies.
      1: float("inf"),
    },
  )

  result = rollout_clips(env, lambda obs: obs["target"], command, torch.tensor([0]))[0]

  assert result.successes == 0
  assert result.nonfinite_failures == 1
  assert result.mpjpe_mm == pytest.approx([5.0])
  assert result.summary()["mpjpe_mm"] == pytest.approx(5.0)
