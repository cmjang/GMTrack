"""Checkpoint-state tests for the GMTrack runner extensions."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from mjlab.rl.runner import MjlabOnPolicyRunner

from gmtrack.mdp.sampling import AdaptiveBinSampler
from gmtrack.rsl_rl.runner import GMTrackOnPolicyRunner
from gmtrack.rsl_rl.storage import TRACKING_FAILURES_EXTRA


def _runner_with_command(command) -> GMTrackOnPolicyRunner:
  runner = object.__new__(GMTrackOnPolicyRunner)
  runner._motion_command = lambda: command  # type: ignore[method-assign]
  return runner


def _sampler(num_bins: int) -> AdaptiveBinSampler:
  return AdaptiveBinSampler(
    clip_ids=torch.tensor([0]),
    clip_bins=torch.tensor([num_bins]),
    max_bins=num_bins,
    num_library_clips=1,
  )


def test_runner_injects_true_terminated_mask_before_ppo_sees_combined_done():
  terminated = torch.tensor([True, False])
  extras = {"time_outs": torch.tensor([True, False])}

  class Wrapper:
    unwrapped = SimpleNamespace(
      termination_manager=SimpleNamespace(terminated=terminated)
    )

    @staticmethod
    def step(_actions):
      return "obs", torch.zeros(2), torch.tensor([1, 0]), extras

  runner = object.__new__(GMTrackOnPolicyRunner)
  runner.env = Wrapper()  # type: ignore[assignment]
  runner._install_tracking_failure_step_hook()
  *_, returned = runner.env.step(torch.zeros(2, 1))

  assert returned[TRACKING_FAILURES_EXTRA].tolist() == [True, False]
  assert returned["time_outs"].tolist() == [True, False]
  terminated.zero_()
  assert returned[TRACKING_FAILURES_EXTRA].tolist() == [True, False]


def test_strict_stage2_checkpoint_must_match_stratification(tmp_path):
  checkpoint = tmp_path / "base.pt"
  checkpoint.write_bytes(b"correct")
  command = SimpleNamespace(
    cfg=SimpleNamespace(require_v1_stratification=True),
    stratification_report={
      "provenance": {"base_checkpoint_sha256": hashlib.sha256(b"correct").hexdigest()}
    },
  )
  runner = _runner_with_command(command)
  runner._validate_stage2_base_checkpoint(
    {"algorithm": {"base_checkpoint": str(checkpoint)}}
  )

  checkpoint.write_bytes(b"wrong")
  with pytest.raises(ValueError, match="does not match"):
    runner._validate_stage2_base_checkpoint(
      {"algorithm": {"base_checkpoint": str(checkpoint)}}
    )


def test_environment_state_round_trip_restores_sampler_not_episode_recovery():
  sampler = _sampler(2)
  sampler.failed_ema.copy_(torch.tensor([[1.0, 2.0]]))
  sampler._pending.copy_(torch.tensor([[3.0, 4.0]]))
  sampler.refresh()
  expected_probs = sampler.probs.clone()
  command = SimpleNamespace(
    sampler_acq=sampler,
    sampler_con=None,
    recovery_mask=torch.tensor([True, False]),
    recovery_assist_raw_n=torch.tensor([42.0, 0.0]),
    recovery_steps_elapsed=123_456,
  )
  runner = _runner_with_command(command)
  saved = runner._collect_env_state()

  sampler.failed_ema.zero_()
  sampler._pending.zero_()
  sampler._probs.fill_(0.5)
  command.recovery_mask.zero_()
  command.recovery_assist_raw_n.zero_()
  command.recovery_steps_elapsed = 0
  runner._restore_env_state(saved)

  assert sampler.failed_ema.tolist() == [[1.0, 2.0]]
  assert sampler._pending.tolist() == [[3.0, 4.0]]
  torch.testing.assert_close(sampler.probs, expected_probs)
  # MuJoCo state is freshly reset on resume, so old episode-local recovery flags
  # must not be overlaid onto it. The assist-force anneal clock, by contrast, is
  # training progress and must survive -- resuming a recovery run must not restart
  # the assistance at full strength.
  assert command.recovery_mask.tolist() == [False, False]
  assert command.recovery_assist_raw_n.tolist() == [0.0, 0.0]
  assert command.recovery_steps_elapsed == 123_456


def test_version_1_state_restarts_the_recovery_anneal_clock():
  """Pre-fix checkpoints hold no recovery-local anneal progress; 0 is the only value.

  Those runs annealed against mjlab's ``common_step_counter``, so the assistance
  force was already pinned at zero. Carrying that forward would keep it there.
  """
  sampler = _sampler(2)
  command = SimpleNamespace(
    sampler_acq=sampler,
    sampler_con=None,
    recovery_steps_elapsed=999,
  )
  runner = _runner_with_command(command)

  runner._restore_env_state(
    {"version": 1, "samplers": {"sampler_acq": sampler.state_dict()}}
  )

  assert command.recovery_steps_elapsed == 0


def test_environment_state_rejects_changed_sampler_shape():
  sampler = _sampler(3)
  command = SimpleNamespace(
    sampler_acq=sampler,
    sampler_con=None,
    recovery_mask=torch.zeros(2, dtype=torch.bool),
    recovery_assist_raw_n=torch.zeros(2),
  )
  runner = _runner_with_command(command)
  incompatible_sampler = _sampler(2).state_dict()
  incompatible = {"version": 1, "samplers": {"sampler_acq": incompatible_sampler}}

  with pytest.raises(ValueError, match="max_bins mismatch"):
    runner._restore_env_state(incompatible)


@pytest.mark.parametrize(
  "saved_samplers",
  [
    {},
    {"sampler_acq": _sampler(2).state_dict(), "sampler_con": _sampler(2).state_dict()},
  ],
)
def test_environment_state_rejects_changed_sampler_layout(saved_samplers):
  command = SimpleNamespace(
    sampler_acq=_sampler(2),
    sampler_con=None,
    recovery_steps_elapsed=0,
  )
  runner = _runner_with_command(command)

  with pytest.raises(ValueError, match="sampler layout does not match"):
    runner._restore_env_state({"version": 2, "samplers": saved_samplers})


@pytest.mark.parametrize(
  ("restore_env_state", "expected_restores"), [(True, 1), (False, 0)]
)
def test_runner_load_can_skip_training_environment_state(
  monkeypatch, restore_env_state, expected_restores
):
  saved = {"version": 2, "samplers": {}}

  def fake_load(_runner, _path, _load_cfg, _strict, _map_location):
    return {"gmtrack_env_state": saved}

  monkeypatch.setattr(MjlabOnPolicyRunner, "load", fake_load)
  runner = object.__new__(GMTrackOnPolicyRunner)
  runner.cfg = {}
  restored = []
  runner._restore_env_state = restored.append  # type: ignore[method-assign]

  infos = runner.load("model.pt", restore_env_state=restore_env_state)

  assert infos == {"gmtrack_env_state": saved}
  assert restored == [saved] * expected_restores
