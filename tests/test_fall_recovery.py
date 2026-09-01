"""Fall-recovery integration (RGMT, arXiv:2601.23080v1, Sec. II-D).

Unit tests exercise the three pieces in isolation -- synthetic randomized fallen
initialization, the annealed upward assistance force, and the 3 s termination shield
-- against lightweight fakes. The integration tests at the bottom verify the same
invariants inside a real environment (CUDA + motion library required).
"""

from __future__ import annotations

from math import pi
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mjlab")

from gmtrack.mdp.commands import MultiMotionCommand
from gmtrack.mdp.events import recovery_assist_force
from gmtrack.mdp.terminations import (
  bad_anchor_pos_z_only,
  bad_anchor_pos_z_only_outside_recovery,
  bad_motion_body_pos_xy,
  bad_motion_body_pos_xy_outside_recovery,
  bad_motion_body_pos_z_only,
  bad_motion_body_pos_z_only_outside_recovery,
)

NUM_ENVS = 8
NUM_JOINTS = 4


def _assist_n(fake) -> torch.Tensor:
  """Evaluate the real annealed-on-read properties against a fake's fields.

  ``SimpleNamespace`` cannot host properties, so the anneal factor is materialized
  onto the fake first and the force property then reads it exactly as it would on a
  real command term.
  """
  fake.recovery_assist_anneal = MultiMotionCommand.recovery_assist_anneal.fget(fake)
  return MultiMotionCommand.recovery_assist_n.fget(fake)


def _fake_command(
  probability: float = 1.0,
  recovery_steps: int = 0,
  anneal_steps: int = 1000,
  acq_env_ids: torch.Tensor | None = None,
):
  """A stand-in with exactly the attributes ``_reset_recovery`` touches."""
  writes: dict[str, torch.Tensor] = {}

  def record_write(env_ids, root_pos, root_ori, lin_vel, ang_vel, joint_pos, joint_vel):
    writes.update(
      env_ids=env_ids,
      root_pos=root_pos,
      root_ori=root_ori,
      lin_vel=lin_vel,
      ang_vel=ang_vel,
      joint_pos=joint_pos,
      joint_vel=joint_vel,
    )

  limits = torch.stack(
    [-torch.ones(NUM_ENVS, NUM_JOINTS), torch.ones(NUM_ENVS, NUM_JOINTS)], dim=-1
  )
  num_frames = 20
  frame_code = torch.arange(num_frames, dtype=torch.float32)
  body_pos_w = torch.zeros(num_frames, 1, 3)
  body_pos_w[:, 0, 2] = 0.1 + 0.1 * frame_code
  angles = 0.2 * frame_code
  body_quat_w = torch.zeros(num_frames, 1, 4)
  body_quat_w[:, 0, 0] = torch.cos(angles / 2)
  body_quat_w[:, 0, 2] = torch.sin(angles / 2)
  body_lin_vel_w = frame_code[:, None, None].repeat(1, 1, 3) + 0.25
  body_ang_vel_w = frame_code[:, None, None].repeat(1, 1, 3) - 0.25
  joint_pos = frame_code[:, None].repeat(1, NUM_JOINTS) / 10
  joint_vel = -joint_pos
  lib = SimpleNamespace(
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    clip_start=torch.tensor([0]),
    clip_len=torch.tensor([num_frames]),
    frame_index=lambda motion_ids, time_steps: time_steps,
  )
  origins = torch.stack(
    [
      torch.arange(NUM_ENVS, dtype=torch.float32),
      torch.zeros(NUM_ENVS),
      torch.zeros(NUM_ENVS),
    ],
    dim=-1,
  )
  fake = SimpleNamespace(
    cfg=SimpleNamespace(
      recovery_probability=probability,
      recovery_assist_force_range=(0.0, 200.0),
      recovery_assist_anneal_steps=anneal_steps,
      recovery_root_height_range=(0.35, 0.65),
      recovery_root_tilt_range=(pi / 3, 2 * pi / 3),
      recovery_joint_position_jitter=(-0.25, 0.25),
    ),
    recovery_mask=torch.zeros(NUM_ENVS, dtype=torch.bool),
    recovery_assist_raw_n=torch.zeros(NUM_ENVS),
    recovery_steps_elapsed=recovery_steps,
    _recovery_anneal_steps=anneal_steps,
    _recovery_window_steps=3,
    motion_ids=torch.zeros(NUM_ENVS, dtype=torch.long),
    time_steps=torch.arange(NUM_ENVS, dtype=torch.long) % num_frames,
    joint_pos=joint_pos[torch.arange(NUM_ENVS) % num_frames].clone(),
    device="cpu",
    acq_env_ids=acq_env_ids if acq_env_ids is not None else torch.arange(NUM_ENVS),
    _env=SimpleNamespace(scene=SimpleNamespace(env_origins=origins)),
    lib=lib,
    robot=SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=limits)),
    _write_reference_state_to_sim=record_write,
  )
  return fake, writes


def test_recovery_init_samples_a_synthetic_fallen_pose_without_repointing_reference():
  torch.manual_seed(0)
  fake, writes = _fake_command(probability=1.0)
  env_ids = torch.arange(NUM_ENVS)
  original_motion_ids = fake.motion_ids.clone()
  original_time_steps = fake.time_steps.clone()

  MultiMotionCommand._reset_recovery(fake, env_ids)

  assert fake.recovery_mask.all()
  assert torch.equal(writes["env_ids"], env_ids)
  assert torch.equal(fake.motion_ids, original_motion_ids)
  assert torch.equal(fake.time_steps, original_time_steps)

  expected_pos = fake._env.scene.env_origins.clone()
  root_z = writes["root_pos"][:, 2] - expected_pos[:, 2]
  assert torch.all((root_z >= 0.35) & (root_z <= 0.65))

  root_quat = writes["root_ori"]
  tilted_z_axis = torch.stack(
    [
      2 * (root_quat[:, 1] * root_quat[:, 3] + root_quat[:, 0] * root_quat[:, 2]),
      2 * (root_quat[:, 2] * root_quat[:, 3] - root_quat[:, 0] * root_quat[:, 1]),
      1 - 2 * (root_quat[:, 1].square() + root_quat[:, 2].square()),
    ],
    dim=-1,
  )
  tilt = torch.acos(torch.clamp(tilted_z_axis[:, 2], -1.0, 1.0))
  assert torch.all((tilt >= pi / 3) & (tilt <= 2 * pi / 3))

  assert not writes["lin_vel"].any()
  assert not writes["ang_vel"].any()
  assert not writes["joint_vel"].any()
  reference_joint_pos = fake.lib.joint_pos[original_time_steps]
  jitter = writes["joint_pos"] - reference_joint_pos
  assert torch.all((jitter >= -0.25) & (jitter <= 0.25))
  assist = _assist_n(fake)
  assert torch.all((assist >= 0.0) & (assist <= 200.0)), "RGMT II-D force range"


def test_recovery_reset_preserves_motion_and_clamps_only_over_late_time():
  torch.manual_seed(0)
  fake, _ = _fake_command(probability=1.0)
  fake.motion_ids = torch.zeros(NUM_ENVS, dtype=torch.long)
  fake.time_steps = torch.arange(NUM_ENVS, dtype=torch.long) + 10
  original_motion_ids = fake.motion_ids.clone()
  original_time_steps = fake.time_steps.clone()

  MultiMotionCommand._reset_recovery(fake, torch.arange(NUM_ENVS))

  assert fake.recovery_mask.all()
  assert torch.equal(fake.motion_ids, original_motion_ids)
  expected_latest = fake.lib.clip_len[0] - fake._recovery_window_steps - 1
  assert torch.equal(
    fake.time_steps, torch.minimum(original_time_steps, expected_latest)
  )
  assert torch.all(
    fake.lib.clip_len[fake.motion_ids] - 1 - fake.time_steps
    >= fake._recovery_window_steps
  )


def test_recovery_disabled_clears_state_and_writes_nothing():
  fake, writes = _fake_command(probability=0.0)
  fake.recovery_mask[:] = True  # stale state from a previous episode
  fake.recovery_assist_raw_n[:] = 50.0

  MultiMotionCommand._reset_recovery(fake, torch.arange(NUM_ENVS))

  assert not fake.recovery_mask.any()
  assert not fake.recovery_assist_raw_n.any()
  assert not writes, "probability 0 must never touch the sim state"


def test_recovery_draw_is_restricted_to_acquisition_envs():
  """Stage II: consolidation rollouts feed the pi_ref alignment loss (Eq. 15), and
  pi_ref never saw fallen states -- so only acquisition envs may recover-init."""
  torch.manual_seed(0)
  acq = torch.arange(NUM_ENVS // 2)  # pace_env_split: acquisition envs come first
  fake, _ = _fake_command(probability=1.0, acq_env_ids=acq)

  MultiMotionCommand._reset_recovery(fake, torch.arange(NUM_ENVS))

  assert fake.recovery_mask[: NUM_ENVS // 2].all()
  assert not fake.recovery_mask[NUM_ENVS // 2 :].any()


def test_assist_force_anneals_linearly_to_zero():
  torch.manual_seed(0)
  halfway, _ = _fake_command(probability=1.0, recovery_steps=500, anneal_steps=1000)
  MultiMotionCommand._reset_recovery(halfway, torch.arange(NUM_ENVS))
  assert torch.all(_assist_n(halfway) <= 100.0), "half anneal caps at 100 N"

  done, _ = _fake_command(probability=1.0, recovery_steps=1000, anneal_steps=1000)
  MultiMotionCommand._reset_recovery(done, torch.arange(NUM_ENVS))
  assert not _assist_n(done).any(), "fully annealed force must be zero"
  # The randomized fallen initialization itself never anneals -- only the force.
  assert done.recovery_mask.all()


def test_assist_anneal_clock_is_recovery_local_not_global_history():
  """Regression: enabling recovery on a resume must not start it fully annealed.

  The anneal used to read mjlab's ``common_step_counter``. Resuming a Stage-I run at
  iteration 40500 restores that counter at 972048 > 720000, so every recovery episode
  drew exactly 0 N -- RGMT Sec. II-D's only exploration aid, silently absent.
  """
  torch.manual_seed(0)
  fresh, _ = _fake_command(probability=1.0, recovery_steps=0, anneal_steps=1000)
  # A long prior training history is simply not visible to the anneal.
  fresh._env.common_step_counter = 10_000_000
  MultiMotionCommand._reset_recovery(fresh, torch.arange(NUM_ENVS))
  assert _assist_n(fresh).max() > 0.0, "assist force must be live at step 0"


def test_applied_assist_follows_a_clock_restored_after_the_episode_was_drawn():
  """Regression: the env is reset before ``runner.load()`` restores the clock.

  ``RslRlVecEnvWrapper.__init__`` resets the environment, which draws each episode's
  recovery assistance, and only afterwards does the runner load the checkpoint. An
  anneal frozen at draw time would hand every first post-resume episode a force
  computed against a clock of zero. Applying the anneal on read makes the restore
  order irrelevant.
  """
  torch.manual_seed(0)
  fake, _ = _fake_command(probability=1.0, recovery_steps=0, anneal_steps=1000)
  MultiMotionCommand._reset_recovery(fake, torch.arange(NUM_ENVS))
  assert _assist_n(fake).max() > 0.0

  fake.recovery_steps_elapsed = 1000  # checkpoint says the anneal is exhausted
  assert not _assist_n(fake).any(), "already-drawn episode ignored the restored clock"

  fake.recovery_steps_elapsed = 750  # ... and tracks it continuously
  assert torch.allclose(_assist_n(fake), 0.25 * fake.recovery_assist_raw_n)


def test_in_recovery_window_respects_mask_and_clock():
  fake = SimpleNamespace(
    recovery_mask=torch.tensor([True, True, False]),
    _recovery_window_steps=150,
    _env=SimpleNamespace(episode_length_buf=torch.tensor([149, 150, 0])),
  )
  window = MultiMotionCommand.in_recovery_window.fget(fake)
  assert window.tolist() == [True, False, False]


def test_record_failures_excludes_recovery_episodes():
  """Recovery failures measure stand-up difficulty, not the bin's tracking
  difficulty; counting them would corrupt c_i in Eq. (12)."""
  recorded = {}

  def record(motion_ids, bins):
    recorded.update(motion_ids=motion_ids, bins=bins)

  fake = SimpleNamespace(
    cfg=SimpleNamespace(sampling_mode="adaptive"),
    _env=SimpleNamespace(
      termination_manager=SimpleNamespace(
        terminated=torch.tensor([True, True, False, False])
      )
    ),
    recovery_mask=torch.tensor([True, False, False, False]),
    motion_ids=torch.tensor([5, 6, 7, 8]),
    time_steps=torch.tensor([10, 20, 30, 40]),
    lib=SimpleNamespace(bin_of=lambda m, t: t // 10),
    sampler_acq=SimpleNamespace(record_failures=record),
    sampler_con=None,
  )

  MultiMotionCommand._record_failures(fake, torch.arange(4))

  # Env 0 terminated but was a recovery episode: only env 1 is attributable.
  assert recorded["motion_ids"].tolist() == [6]
  assert recorded["bins"].tolist() == [2]


def _termination_env(
  in_window: torch.Tensor, robot_anchor_z: float = 0.3
) -> SimpleNamespace:
  n = int(in_window.numel())
  command = SimpleNamespace(
    anchor_pos_w=torch.tensor([[0.0, 0.0, 0.8]]).repeat(n, 1),
    robot_anchor_pos_w=torch.tensor([[0.0, 0.0, robot_anchor_z]]).repeat(n, 1),
    in_recovery_window=in_window,
  )
  return SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _name: command)
  )


def test_shielded_termination_suppressed_only_inside_window():
  # Robot anchor 0.5 m below the reference everywhere: the base check fires for
  # every env, so any False below is the shield's doing.
  env = _termination_env(torch.tensor([True, False, False]))
  base = bad_anchor_pos_z_only(env, "motion", threshold=0.25)
  gated = bad_anchor_pos_z_only_outside_recovery(env, "motion", threshold=0.25)
  assert base.tolist() == [True, True, True]
  assert gated.tolist() == [False, True, True]


def test_foot_xy_termination_uses_reanchored_positions_and_recovery_shield():
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=("left_ankle", "right_ankle", "wrist")),
    # These references are already pelvis-XY/yaw-aligned by MultiMotionCommand. The
    # first ankle exceeds the XY threshold; the second only has a large Z error,
    # which must not leak into the horizontal placement check.
    body_pos_relative_w=torch.tensor(
      [
        [[0.21, 0.0, 0.0], [0.0, 0.19, 0.5], [0.0, 0.0, 0.5]],
        [[0.21, 0.0, 0.0], [0.0, 0.19, 0.5], [0.0, 0.0, 0.5]],
      ]
    ),
    robot_body_pos_w=torch.zeros(2, 3, 3),
    in_recovery_window=torch.tensor([True, False]),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))
  feet = ("left_ankle", "right_ankle")

  assert bad_motion_body_pos_xy(env, "motion", 0.2, feet).tolist() == [True, True]
  assert bad_motion_body_pos_xy_outside_recovery(env, "motion", 0.2, feet).tolist() == [
    False,
    True,
  ]


def test_foot_z_termination_is_independent_from_xy_and_uses_015_threshold():
  command = SimpleNamespace(
    cfg=SimpleNamespace(body_names=("left_ankle", "right_ankle")),
    body_pos_relative_w=torch.tensor(
      [
        [[0.4, 0.0, 0.14], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.16], [0.0, 0.0, 0.0]],
      ]
    ),
    robot_body_pos_w=torch.zeros(2, 2, 3),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))

  # A large horizontal error alone does not trip Z; 0.16 m does trip 0.15 m.
  assert bad_motion_body_pos_z_only(env, "motion", 0.15).tolist() == [False, True]


def test_assist_force_event_writes_upward_wrench_for_active_envs_only():
  calls = {}

  class Asset:
    def write_external_wrench_to_sim(self, forces, torques, body_ids):
      calls.update(forces=forces, torques=torques, body_ids=body_ids)

  command = SimpleNamespace(
    in_recovery_window=torch.tensor([True, False, True]),
    recovery_assist_n=torch.tensor([120.0, 80.0, 0.0]),
    robot_anchor_body_index=7,
  )
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    scene={"robot": Asset()},
    command_manager=SimpleNamespace(get_term=lambda _name: command),
  )

  recovery_assist_force(env, None, "motion", SimpleNamespace(name="robot"))

  forces = calls["forces"]
  assert forces.shape == (3, 1, 3)
  # Upward (+z) with the per-episode magnitude; env 1 is outside the window and env
  # 2's annealed magnitude is zero -- both must be written as zeros (xfrc persists).
  assert forces[:, 0, 2].tolist() == [120.0, 0.0, 0.0]
  assert not forces[:, 0, :2].any(), "assist is purely vertical"
  assert not calls["torques"].any()
  assert calls["body_ids"] == [7]


def test_env_cfg_keeps_recovery_as_an_explicit_proxy():
  from gmtrack import _stage1_env, _stage2_env
  from gmtrack.envs.env_cfg import RECOVERY_PROBABILITY, make_gmtrack_env_cfg

  for cfg in (_stage1_env(), _stage2_env()):
    motion = cfg.commands["motion"]
    assert RECOVERY_PROBABILITY == 0.15  # RGMT II-D proxy value
    assert motion.recovery_probability == 0.0
    assert motion.recovery_window_s == 3.0  # RGMT II-D
    assert motion.recovery_assist_force_range == (0.0, 200.0)  # RGMT II-D
    assert motion.recovery_root_height_range == (0.35, 0.65)
    assert motion.recovery_root_tilt_range == (pi / 3, 2 * pi / 3)
    assert motion.recovery_joint_position_jitter == (-0.25, 0.25)
    assert "recovery_assist" not in cfg.events
    assert "recovery_assist" not in cfg.observations["critic"].terms
    assert cfg.terminations["anchor_pos"].func is bad_anchor_pos_z_only_outside_recovery
    assert (
      cfg.terminations["ee_body_pos"].func
      is bad_motion_body_pos_z_only_outside_recovery
    )
    assert "foot_pos_xy" not in cfg.terminations
    assert "foot_pos_z" not in cfg.terminations

  proxy = make_gmtrack_env_cfg(
    manifest=_stage1_env().commands["motion"].manifest,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  assert proxy.commands["motion"].recovery_probability == 0.15
  assert proxy.events["recovery_assist"].mode == "step"
  assert "recovery_assist" not in proxy.observations["critic"].terms
  assert "recovery_assist" not in proxy.observations["proprio_hist"].terms

  play = _stage1_env(play=True)
  assert play.commands["motion"].recovery_probability == 0.0
  assert "recovery_assist" not in play.events


def test_recovery_requires_config_construction_not_a_cli_flag():
  """`recovery_probability` gates term *construction*, so raising it on an
  already-built strict config yields fallen resets and a termination shield with no
  assistance force at all -- RGMT's only mechanism for escaping fallen states."""
  from gmtrack import _stage1_env
  from gmtrack.envs.env_cfg import RECOVERY_PROBABILITY, make_gmtrack_env_cfg

  manifest = _stage1_env().commands["motion"].manifest
  recovery = make_gmtrack_env_cfg(
    manifest=manifest,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  assert recovery.commands["motion"].recovery_probability == 0.15
  assert "recovery_assist" in recovery.events
  assert "recovery_assist" not in recovery.observations["critic"].terms
  play = make_gmtrack_env_cfg(
    manifest=manifest,
    recovery_probability=RECOVERY_PROBABILITY,
    play=True,
  )
  assert play.commands["motion"].recovery_probability == 0.0

  for heading_closed_loop in (False, True):
    train = make_gmtrack_env_cfg(
      manifest=manifest,
      causal_online=True,
      heading_closed_loop=heading_closed_loop,
      sonic_foot_terminations=True,
      recovery_probability=RECOVERY_PROBABILITY,
    )
    assert train.commands["motion"].recovery_probability == 0.15
    assert "recovery_assist" in train.events
    assert {"foot_pos_xy", "foot_pos_z"} <= set(train.terminations)

  # The broken CLI path must now fail loudly instead of training without the force.
  faked_cli = _stage1_env()
  faked_cli.commands["motion"].recovery_probability = 0.15
  assert "recovery_assist" not in faked_cli.events


# -- integration (CUDA + motion library) --------------------------------------

_MANIFEST = Path(
  "data/current/"
  "stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"
)

needs_runtime = pytest.mark.skipif(
  not torch.cuda.is_available() or not _MANIFEST.exists(),
  reason="needs CUDA and a motion library (run prepare_motions)",
)


@pytest.fixture(scope="module")
def recovery_env():
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv

  import gmtrack  # noqa: F401
  from gmtrack.envs.env_cfg import make_gmtrack_env_cfg

  cfg = make_gmtrack_env_cfg(
    manifest=str(_MANIFEST),
    recovery_probability=1.0,
  )
  cfg.scene.num_envs = 2
  cfg.commands["motion"].recovery_probability = 1.0  # every reset is a recovery
  e = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  yield e
  e.close()


def _zero_step(env):
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  return env.step(action)


@needs_runtime
def test_assist_force_is_live_and_its_clock_advances_one_step_per_env_step(
  recovery_env,
):
  """Runtime counterpart to the unit test: the force must actually reach the sim.

  The anneal clock has to advance exactly once per environment step, and at the
  start of recovery training the assistance must be at (nearly) full strength --
  the failure this replaces had it pinned at 0 N for an entire 33k-iteration run.
  """
  env = recovery_env
  command = env.command_manager.get_term("motion")
  command.recovery_steps_elapsed = 0
  env.reset()

  assert command.recovery_mask.all()
  assert command.recovery_assist_n.max() > 0.0, "assist force annealed away at step 0"

  # ManagerBasedRlEnv.reset also runs command_manager.compute (with dt=0), so a
  # counter kept inside _update_command would already read 1 here.
  assert command.recovery_steps_elapsed == 0, "reset must not tick the anneal clock"
  for _ in range(5):
    _zero_step(env)
  assert command.recovery_steps_elapsed == 5

  # xfrc_applied is indexed by *global* body id; the command holds an entity-local one.
  robot = env.scene["robot"]
  anchor = robot.data.indexing.body_ids[command.robot_anchor_body_index]
  applied = env.sim.data.xfrc_applied[:, anchor]
  in_window = command.in_recovery_window
  assert in_window.any(), "test setup: the 3 s window closed before the assertion"
  torch.testing.assert_close(
    applied[in_window, 2], command.recovery_assist_n[in_window]
  )
  assert not applied[:, :2].any(), "assistance must be purely vertical"


@needs_runtime
def test_recovery_reset_starts_from_a_fallen_pose(recovery_env):
  env = recovery_env
  env.reset()
  command = env.command_manager.get_term("motion")

  assert command.recovery_mask.all()
  root_z = command.robot_body_pos_w[:, 0, 2] - env.scene.env_origins[:, 2]
  assert torch.all(
    (root_z >= command.cfg.recovery_root_height_range[0])
    & (root_z <= command.cfg.recovery_root_height_range[1])
  )
  root_quat = command.robot_body_quat_w[:, 0]
  tilted_z_axis = torch.stack(
    [
      2 * (root_quat[:, 1] * root_quat[:, 3] + root_quat[:, 0] * root_quat[:, 2]),
      2 * (root_quat[:, 2] * root_quat[:, 3] - root_quat[:, 0] * root_quat[:, 1]),
      1 - 2 * (root_quat[:, 1].square() + root_quat[:, 2].square()),
    ],
    dim=-1,
  )
  tilt = torch.acos(torch.clamp(tilted_z_axis[:, 2], -1.0, 1.0))
  assert torch.all(
    (tilt >= command.cfg.recovery_root_tilt_range[0])
    & (tilt <= command.cfg.recovery_root_tilt_range[1])
  )
  assert not command.robot_anchor_lin_vel_w.any()
  assert not command.robot_anchor_ang_vel_w.any()
  assert not command.robot_joint_vel.any()
  assert torch.all(
    (command.recovery_assist_n >= 0.0) & (command.recovery_assist_n <= 200.0)
  )


@needs_runtime
def test_assist_wrench_reaches_the_anchor_body(recovery_env):
  env = recovery_env
  env.reset()
  _zero_step(env)
  command = env.command_manager.get_term("motion")
  wrench = env.scene["robot"].data.body_external_wrench

  active = command.in_recovery_window
  anchor = command.robot_anchor_body_index
  assert torch.allclose(wrench[active, anchor, 2], command.recovery_assist_n[active])
  # Purely vertical force, no torque, nothing on other bodies.
  assert not wrench[:, anchor, :2].any()
  assert not wrench[:, anchor, 3:].any()
  other = [i for i in range(wrench.shape[1]) if i != anchor]
  assert not wrench[:, other, :].any()


@needs_runtime
def test_shield_holds_through_window_then_terminates_unrecovered(recovery_env):
  """The end-to-end RGMT II-D contract: a robot dumped on the floor with zero
  residual actions must survive exactly the recovery window, then be terminated by
  the re-engaged instability checks."""
  env = recovery_env
  env.reset()
  command = env.command_manager.get_term("motion")
  window = command._recovery_window_steps
  assert window == 150  # 3 s at 50 Hz (RGMT II-D)

  shield_violations = 0
  terminated_after_window = False
  base_check_fired = False
  for _ in range(window + 20):
    # Terminations are evaluated inside step() *before* auto-reset zeroes the
    # episode clock, so the window membership at check time derives from the
    # pre-step counter (+1 for the step being taken), not the post-step one.
    buf_at_check = env.episode_length_buf + 1
    mask_at_check = command.recovery_mask.clone()
    _zero_step(env)
    terminated = env.termination_manager.terminated
    in_window = buf_at_check < window
    # Fallen robot vs. upright reference: the *base* check fires long before the
    # window ends; the manager-level (shielded) term must stay quiet in-window.
    base_check_fired |= bool(
      bad_motion_body_pos_z_only(env, "motion", 0.25, None).any()
    )
    shield_violations += int((terminated & mask_at_check & in_window).sum())
    terminated_after_window |= bool((terminated & ~in_window).any())

  assert base_check_fired, "test setup: the fallen pose never tripped the base check"
  assert shield_violations == 0, "instability termination fired inside the window"
  assert terminated_after_window, "no termination after the window elapsed"
