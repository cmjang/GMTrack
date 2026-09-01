"""Evaluation-harness invariants: paper-criterion rollouts (Sec. VI-A)."""

from dataclasses import dataclass

import pytest

from gmtrack.envs.env_cfg import make_gmtrack_env_cfg
from gmtrack.scripts._harness import (
  _configure_random_recovery_rollout,
  _configure_rollout_env_cfg,
  _configure_rollout_motion,
  _inference_runner_cfg,
  _resolve_eval_mode,
  strip_failure_terminations,
)


def test_strip_removes_exactly_the_failure_terminations():
  cfg = make_gmtrack_env_cfg(manifest="unused.json")
  assert set(cfg.terminations) == {
    "nonfinite_physics_state",
    "time_out",
    "motion_sequence_end",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
  }
  strip_failure_terminations(cfg)
  # The harness owns sequence length too, so no command-driven reset can destroy
  # the final physics state before it is scored. The training nonfinite guard stays
  # disabled because rollout_eval diagnoses integrity failures without auto-reset.
  assert set(cfg.terminations) == {"time_out"}
  assert all(term.time_out for term in cfg.terminations.values())


def test_strip_removes_optional_causal_foot_terminations():
  cfg = make_gmtrack_env_cfg(manifest="unused.json", sonic_foot_terminations=True)
  assert {"foot_pos_xy", "foot_pos_z"} <= set(cfg.terminations)
  strip_failure_terminations(cfg)
  assert set(cfg.terminations) == {"time_out"}


def test_strip_raises_on_missing_or_unknown_failure_terms():
  cfg = make_gmtrack_env_cfg(manifest="unused.json")
  strip_failure_terminations(cfg)
  # Applying twice must fail loudly, not silently no-op.
  with pytest.raises(KeyError):
    strip_failure_terminations(cfg)

  cfg = make_gmtrack_env_cfg(manifest="unused.json")
  cfg.terminations["novel_failure"] = cfg.terminations["ee_body_pos"]
  with pytest.raises(ValueError, match="novel_failure"):
    strip_failure_terminations(cfg)


def test_rollout_configuration_disables_auto_reset():
  cfg = make_gmtrack_env_cfg(manifest="unused.json")
  _configure_rollout_env_cfg(cfg)
  assert cfg.auto_reset is False
  assert cfg.episode_length_s == int(1e9)
  assert set(cfg.terminations) == {"time_out"}


def test_rollout_motion_clears_stage2_roles_and_applies_subset_manifest():
  cfg = make_gmtrack_env_cfg(
    manifest="full.json",
    acquisition_fraction=0.8,
    acquisition_clips="challenging.json",
    consolidation_clips="mastered.json",
    require_v1_stratification=True,
  )
  _configure_rollout_motion(cfg, "evaluation-subset.json")

  motion = cfg.commands["motion"]
  assert motion.manifest == "evaluation-subset.json"
  assert motion.acquisition_fraction is None
  assert motion.acquisition_clips is None
  assert motion.consolidation_clips is None
  assert motion.require_v1_stratification is False
  assert motion.sampling_mode == "start"
  assert motion.clamp_at_end is True
  assert motion.recovery_probability == 0.0
  for group in cfg.observations.values():
    for term in group.terms.values():
      if "acquisition_fraction" in term.params:
        assert term.params["acquisition_fraction"] is None
  for event in cfg.events.values():
    if "acquisition_fraction" in event.params:
      assert event.params["acquisition_fraction"] is None


def test_random_recovery_rollout_is_unassisted_and_nominal():
  cfg = make_gmtrack_env_cfg(manifest="unused.json", recovery_probability=0.15)
  # The RGMT recovery rollout uses synthetic fallen poses and exposes no recovery
  # clip selector at all.
  assert not hasattr(cfg.commands["motion"], "recovery_clip_name_patterns")
  _configure_rollout_motion(cfg, None)
  _configure_random_recovery_rollout(cfg, nominal=True)

  motion = cfg.commands["motion"]
  assert motion.recovery_probability == 1.0
  assert motion.recovery_assist_force_range == (0.0, 0.0)
  assert motion.pose_range == {}
  assert motion.velocity_range == {}
  assert motion.joint_position_range == (0.0, 0.0)
  assert set(cfg.events) == {"recovery_assist"}
  assert "recovery_assist" not in cfg.observations["critic"].terms
  for group in cfg.observations.values():
    for term in group.terms.values():
      if "enabled" in term.params:
        assert term.params["enabled"] is False


def test_random_recovery_rollout_requires_recovery_task():
  cfg = make_gmtrack_env_cfg(manifest="unused.json")
  with pytest.raises(ValueError, match="requires a task constructed with recovery"):
    _configure_random_recovery_rollout(cfg, nominal=True)


def test_explicit_and_legacy_eval_modes_must_agree():
  assert _resolve_eval_mode(play=None, eval_mode=None) == "nominal"
  assert _resolve_eval_mode(play=True, eval_mode=None) == "nominal"
  assert _resolve_eval_mode(play=False, eval_mode=None) == "randomized"
  assert _resolve_eval_mode(play=None, eval_mode="randomized") == "randomized"
  with pytest.raises(ValueError, match="Conflicting"):
    _resolve_eval_mode(play=True, eval_mode="randomized")


@dataclass
class _FakeAgentCfg:
  algorithm: dict


def test_inference_runner_does_not_require_stage1_base_checkpoint():
  cfg = _FakeAgentCfg(
    algorithm={
      "base_checkpoint": "/missing/stage1.pt",
      "consolidation_enabled": True,
      "acquisition_fraction": 0.8,
    }
  )
  inference_cfg = _inference_runner_cfg(cfg)
  assert inference_cfg["algorithm"]["base_checkpoint"] is None
  assert inference_cfg["algorithm"]["acquisition_fraction"] is None
  assert inference_cfg["algorithm"]["consolidation_enabled"] is False
  assert inference_cfg["algorithm"]["use_star"] is False
  assert cfg.algorithm["base_checkpoint"] == "/missing/stage1.pt"
