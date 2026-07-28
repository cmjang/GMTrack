"""Config-level checks against the paper's tables.

These catch the class of error that never raises: a reward weight that drifted, a
noise vector whose channels no longer line up with the observation it decorates, or a
policy rate that stopped matching the motion clips' fps.
"""

from __future__ import annotations

import pytest

from ex_grmt.envs.env_cfg import (
  COMMAND_WINDOW_NOISE,
  COMMAND_WINDOW_RADIUS,
  DEFAULT_SIM_HZ,
  HISTORY_LENGTH,
  POLICY_HZ,
)
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
