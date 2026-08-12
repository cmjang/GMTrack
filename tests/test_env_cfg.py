"""Config-level checks against the paper's tables.

These catch the class of error that never raises: a reward weight that drifted, a
noise vector whose channels no longer line up with the observation it decorates, or a
policy rate that stopped matching the motion clips' fps.
"""

from __future__ import annotations

from math import pi
from types import SimpleNamespace

import pytest
import torch

from ex_grmt import (
  MANIFEST_CHALLENGING,
  MANIFEST_MASTERED,
  MANIFEST_STAGE1,
  MANIFEST_STRATIFIED,
  _challenging_only_env,
  _mixed_env,
  _stage1_env,
  _stage2_env,
)
from ex_grmt.envs.env_cfg import (
  COMMAND_WINDOW_NOISE,
  COMMAND_WINDOW_RADIUS,
  DEFAULT_SIM_HZ,
  HISTORY_LENGTH,
  POLICY_HZ,
)
from ex_grmt.mdp import events as ex_events
from ex_grmt.mdp.events import _acquisition_env_ids
from ex_grmt.mdp.observations import _add_acquisition_uniform_noise
from ex_grmt.rl_cfgs import (
  ACQUISITION_FRACTION,
  BETA,
  KAPPA,
  LAMBDA_BASE,
  RHO_REF,
  RHO_STAR,
  RHO_TOPK,
  stage1_runner_cfg,
  stage2_runner_cfg,
)
from ex_grmt.rsl_rl.config import ExGRMTActorCfg

TOKEN_DIM = 38  # 3 + 3 + 3 + 29 (Eq. 2)
NUM_TOKENS = 21  # 2L + 1, L = 10


def test_command_window_noise_matches_table_ii_per_channel():
  assert len(COMMAND_WINDOW_NOISE) == NUM_TOKENS * TOKEN_DIM
  token = COMMAND_WINDOW_NOISE[:TOKEN_DIM]
  assert token[0:3] == (0.5,) * 3, "base linear velocity"
  assert token[3:6] == (0.52,) * 3, "base angular velocity"
  assert token[6:9] == (0.05,) * 3, "gravity direction"
  assert token[9:38] == (0.1,) * 29, "joint pose"
  # Same pattern in every token.
  for i in range(NUM_TOKENS):
    assert COMMAND_WINDOW_NOISE[i * TOKEN_DIM : (i + 1) * TOKEN_DIM] == token


def test_window_radius_gives_21_tokens():
  assert 2 * COMMAND_WINDOW_RADIUS + 1 == NUM_TOKENS


def test_policy_rate_divides_sim_rate():
  """The policy rate must equal the clips' fps, or reference and control desync."""
  assert POLICY_HZ == 50.0
  ratio = DEFAULT_SIM_HZ / POLICY_HZ
  assert ratio == pytest.approx(round(ratio))


def test_collision_solver_capacity_covers_randomized_recovery_contacts():
  """Dense fallen poses must not overflow and poison observations with NaNs."""
  sim = _stage1_env().sim
  assert sim.njmax >= 512
  assert sim.contact_sensor_maxmatch >= 128


def test_nonfinite_physics_state_is_a_terminal_failure():
  term = _stage1_env().terminations["nonfinite_physics_state"]
  assert term.time_out is False


def test_actor_dims_match_table_iii():
  cfg = ExGRMTActorCfg()
  assert cfg.hidden_dims == (1024, 1024, 512, 256)
  assert cfg.token_dim == 64
  assert cfg.history_length == HISTORY_LENGTH == 10
  assert sum(cfg.proprio_term_dims) == 64, "o^prop is 64-d (Eq. 1)"
  assert cfg.fsq_token_dim == 32, "u_t factorizes into two 32-d tokens (Eq. 10)"


def test_pace_and_star_constants_match_the_paper():
  assert (ACQUISITION_FRACTION, LAMBDA_BASE, KAPPA, RHO_REF, BETA) == (
    0.8,
    0.3,
    5.0,
    0.6,
    0.99,
  )
  assert (RHO_TOPK, RHO_STAR) == (0.05, 0.25)


def test_ppo_hyperparameters_match_table_iii():
  alg = stage1_runner_cfg().algorithm
  assert alg.num_learning_epochs == 5
  assert alg.num_mini_batches == 4
  assert alg.learning_rate == 1e-3
  assert alg.schedule == "adaptive"
  assert alg.desired_kl == 0.01
  assert alg.gamma == 0.99
  assert alg.lam == 0.95
  assert alg.clip_param == 0.2
  assert alg.entropy_coef == 0.005
  assert stage1_runner_cfg().num_steps_per_env == 24


def test_stage1_disables_pace_and_star():
  """Stage I is plain PPO with adaptive sampling -- no roles, no resampling."""
  alg = stage1_runner_cfg().algorithm
  assert alg.acquisition_fraction is None
  assert alg.consolidation_enabled is False
  assert alg.use_star is False


def test_stage2_enables_both():
  alg = stage2_runner_cfg().algorithm
  assert alg.acquisition_fraction == ACQUISITION_FRACTION
  assert alg.consolidation_enabled is True
  assert alg.use_star is True


def test_strict_v1_tasks_disable_recovery_proxy():
  for cfg in (_stage1_env(), _stage2_env()):
    motion = cfg.commands["motion"]
    assert motion.recovery_probability == 0.0
    assert "recovery_assist" not in cfg.events

  from ex_grmt.envs.env_cfg import RECOVERY_PROBABILITY, make_ex_grmt_env_cfg

  stage2 = _stage2_env()
  proxy = make_ex_grmt_env_cfg(
    manifest=MANIFEST_STRATIFIED,
    acquisition_clips=stage2.commands["motion"].acquisition_clips,
    consolidation_clips=stage2.commands["motion"].consolidation_clips,
    acquisition_fraction=0.8,
    recovery_probability=RECOVERY_PROBABILITY,
  )
  motion = proxy.commands["motion"]
  assert motion.recovery_probability == 0.15
  assert motion.recovery_root_height_range == (0.35, 0.65)
  assert motion.recovery_root_tilt_range == (pi / 3, 2 * pi / 3)
  assert motion.recovery_joint_position_jitter == (-0.25, 0.25)
  assert "recovery_assist" in proxy.events
  assert "recovery_assist" not in proxy.observations["critic"].terms


def test_stage_manifests_and_episode_boundaries_follow_paper_order():
  stage1 = _stage1_env()
  stage2 = _stage2_env()
  assert stage1.commands["motion"].manifest == MANIFEST_STAGE1
  assert stage2.commands["motion"].manifest == MANIFEST_STRATIFIED
  assert stage2.commands["motion"].require_v1_stratification is True
  assert stage1.episode_length_s == 10.0
  assert stage1.terminations["motion_sequence_end"].time_out is True


def test_strict_ablation_validation_is_independent_of_sampling_pool():
  finetune = _challenging_only_env().commands["motion"]
  mixed = _mixed_env().commands["motion"]

  for motion in (finetune, mixed):
    assert motion.require_v1_stratification is True
    assert motion.stratification_mastered_manifest == MANIFEST_MASTERED
    assert motion.stratification_challenging_manifest == MANIFEST_CHALLENGING
  assert finetune.acquisition_clips == MANIFEST_CHALLENGING
  assert finetune.acquisition_fraction is None
  assert mixed.acquisition_clips is None
  assert mixed.consolidation_clips is None


def test_stage2_randomization_is_role_masked_and_nominal_play_is_clean():
  stage2 = _stage2_env()
  for event in stage2.events.values():
    if "acquisition_fraction" in event.params:
      assert event.params["acquisition_fraction"] == 0.8

  motion = stage2.commands["motion"]
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == (0.0, 0.0)

  nominal = _stage2_env(play=True)
  assert nominal.events == {}
  assert nominal.commands["motion"].joint_position_range == (0.0, 0.0)


def test_role_mask_keeps_consolidation_observations_clean():
  torch.manual_seed(0)
  clean = torch.zeros(10, 3)
  noisy = _add_acquisition_uniform_noise(clean, 0.5, 0.8, enabled=True)
  assert bool((noisy[:8] != 0.0).any())
  assert torch.equal(noisy[8:], clean[8:])

  env = SimpleNamespace(num_envs=10, device="cpu")
  requested = torch.tensor([1, 7, 8, 9])
  assert _acquisition_env_ids(env, requested, 0.8).tolist() == [1, 7]


def test_ground_friction_covers_all_collision_geometries():
  cfg = _stage1_env()
  floor = cfg.events["terrain_friction_floor"]
  assert floor.params["ranges"] == (0.10, 0.10)
  assert floor.params["asset_cfg"].name == "terrain"
  assert floor.params["asset_cfg"].geom_names == ("terrain",)
  asset_cfg = cfg.events["ground_friction"].params["asset_cfg"]
  assert asset_cfg.geom_names == r".*_collision\d*$"


@pytest.mark.parametrize(
  ("wrapper_name", "upstream_name"),
  (
    ("role_geom_friction", "geom_friction"),
    ("role_body_mass", "body_mass"),
    ("role_body_com_offset", "body_com_offset"),
    ("role_effort_limits", "effort_limits"),
    ("role_pd_gains", "pd_gains"),
    ("role_joint_armature", "joint_armature"),
  ),
)
def test_role_dr_wrappers_preserve_model_field_expansion(wrapper_name, upstream_name):
  """Wrapping an event must retain mjlab's per-world expansion/recompute metadata."""
  from mjlab.envs.mdp import dr

  wrapper = getattr(ex_events, wrapper_name)
  upstream = getattr(dr, upstream_name)
  assert wrapper.model_fields == upstream.model_fields
  assert wrapper.recompute == upstream.recompute


def test_feet_and_wrists_may_touch_the_ground_unpenalized():
  """Table I never defines the "undesired" body set; we follow InstinctLab, which
  exempts exactly ankles + wrists so get-ups and hand-support motions are not
  penalized for doing what their reference demands."""
  import re

  cfg = _stage1_env()
  sensor = next(s for s in cfg.scene.sensors if s.name == "nonfoot_ground_contact")
  excludes = sensor.primary.exclude
  assert len(excludes) == 2
  for geom in ("left_foot3_collision", "right_foot7_collision"):
    assert any(re.match(rx, geom) for rx in excludes), geom
  for geom in ("left_hand_collision", "right_hand_collision"):
    assert any(re.match(rx, geom) for rx in excludes), geom
  # Elbows and knees remain penalized -- InstinctLab's set, not SONIC's.
  for geom in ("left_elbow_collision", "right_knee_collision", "pelvis_collision"):
    assert not any(re.match(rx, geom) for rx in excludes), geom


def test_critic_is_wider_than_actor_per_table_iii():
  cfg = stage1_runner_cfg()
  assert cfg.critic.hidden_dims == (1024, 1024, 512, 512)
  assert cfg.actor.hidden_dims == (1024, 1024, 512, 256)
  assert cfg.actor.obs_normalization is False, "LayerNorm replaces it (Sec. IV-A)"
  assert cfg.critic.obs_normalization is True


def test_star_group_is_not_a_model_input():
  """The `star` group carries STAR metadata into storage; no network may see it."""
  groups = stage1_runner_cfg().obs_groups
  for name, entries in groups.items():
    assert "star" not in entries, f"'star' leaked into obs_groups[{name!r}]"
