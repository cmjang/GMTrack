"""Build a trained-policy environment outside mjlab's train/play CLIs.

`stratify.py` and `evaluate.py` both need "env + loaded policy, no viewer, no logging".
mjlab's `play.py` does this inline, tangled with W&B artifact resolution and viewer
setup, so the few lines that matter are lifted here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal, cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from gmtrack.mdp.commands import MultiMotionCommand


def strip_failure_terminations(env_cfg) -> None:
  """Remove training terminations from paper-criterion rollouts.

  Paper Sec. VI-A: a rollout runs "until the reference terminates or the policy
  fails", and failure is judged solely by the root-height criterion applied in
  ``rollout_eval``. The training failure terminations (most easily the end-effector
  z-deviation and optional foot XY/Z checks) trip well before that criterion and
  would deflate Succ. below anything comparable to Table VI. Sequence completion is
  owned by the rollout harness as an exact ``N - 1`` transition horizon: retaining
  that timeout would let the environment reset before its terminal state is measured.
  ``del`` raises if a name is ever missing, so a renamed termination cannot silently
  enter evaluation.

  ``nonfinite_physics_state`` is removed too: ``rollout_eval`` diagnoses non-finite
  state as a data-integrity failure while retaining the finite metric prefix. It is
  not a training termination or an alternate paper success criterion here.
  """
  for term in (
    "nonfinite_physics_state",
    "motion_sequence_end",
    "anchor_pos",
    "anchor_ori",
    "ee_body_pos",
  ):
    del env_cfg.terminations[term]
  # These two guards exist only on the opt-in causal Stage-I variants.
  env_cfg.terminations.pop("foot_pos_xy", None)
  env_cfg.terminations.pop("foot_pos_z", None)
  for name, cfg in env_cfg.terminations.items():
    if not cfg.time_out:
      raise ValueError(
        f"Termination '{name}' is a failure termination unknown to "
        "strip_failure_terminations; add it there or evaluation will end rollouts "
        "before the paper's success criterion does."
      )


def _resolve_eval_mode(
  *, play: bool | None, eval_mode: Literal["nominal", "randomized"] | None
) -> Literal["nominal", "randomized"]:
  """Resolve the old ``play`` switch into an explicit rollout mode."""
  legacy_mode = None if play is None else ("nominal" if play else "randomized")
  if eval_mode is not None and legacy_mode is not None and eval_mode != legacy_mode:
    raise ValueError(
      f"Conflicting rollout modes: play={play} means {legacy_mode!r}, but "
      f"eval_mode={eval_mode!r}."
    )
  return eval_mode or legacy_mode or "nominal"


def _configure_rollout_env_cfg(env_cfg) -> None:
  """Make terminal physics observable and hand the exact horizon to the harness."""
  env_cfg.auto_reset = False
  env_cfg.episode_length_s = int(1e9)
  strip_failure_terminations(env_cfg)


def _configure_rollout_motion(env_cfg, manifest: str | None) -> None:
  """Disable training-only PACE roles while retaining the requested eval mode.

  Evaluation pins every clip explicitly. Keeping Stage-II's mastered/challenging
  split would both resample underneath the harness and try to resolve clip names
  that are absent when ``manifest`` is a subset. In randomized mode every rollout
  should receive the perturbations, rather than only the old acquisition prefix.
  """
  motion_cfg = env_cfg.commands["motion"]
  if manifest is not None:
    motion_cfg.manifest = manifest
  motion_cfg.acquisition_fraction = None
  motion_cfg.acquisition_clips = None
  motion_cfg.consolidation_clips = None
  # Evaluation validates the requested subset and its complete sibling artifact set
  # before calling this harness. The environment itself then loads only that subset,
  # so its training-only three-manifest validation must not run a second time here.
  # Stratification also uses this path with its explicitly marked provisional input.
  motion_cfg.require_v1_stratification = False
  motion_cfg.sampling_mode = "start"
  motion_cfg.clamp_at_end = True
  motion_cfg.recovery_probability = 0.0

  for group_cfg in env_cfg.observations.values():
    for term_cfg in group_cfg.terms.values():
      if "acquisition_fraction" in term_cfg.params:
        term_cfg.params["acquisition_fraction"] = None
  for event_cfg in env_cfg.events.values():
    if "acquisition_fraction" in event_cfg.params:
      event_cfg.params["acquisition_fraction"] = None


def _configure_random_recovery_rollout(env_cfg, *, nominal: bool) -> None:
  """Force every reset to use an unassisted randomized unstable pose.

  Recovery has to be present when the task config is constructed because it adds an
  event. Web visualization loads that construction, then this helper turns the
  training probability into one and removes the upward exploration force. In nominal
  mode the remaining Table-II perturbations and observation corruption are disabled
  too, so the viewer isolates recovery skill.
  """
  if "recovery_assist" not in env_cfg.events:
    raise ValueError(
      "Random recovery visualization requires an environment config constructed "
      "with recovery enabled."
    )

  motion_cfg = env_cfg.commands["motion"]
  motion_cfg.recovery_probability = 1.0
  motion_cfg.recovery_assist_force_range = (0.0, 0.0)
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0.0, 0.0)

  if not nominal:
    return

  for group_cfg in env_cfg.observations.values():
    for term_cfg in group_cfg.terms.values():
      if "enabled" in term_cfg.params:
        term_cfg.params["enabled"] = False
  for event_name in tuple(env_cfg.events):
    if event_name != "recovery_assist":
      env_cfg.events.pop(event_name)


def _inference_runner_cfg(agent_cfg) -> dict:
  """Strip Stage-II-only training state before constructing an inference runner."""
  runner_cfg = asdict(agent_cfg)
  algorithm_cfg = runner_cfg["algorithm"]
  # A viewer/evaluator only restores the actor.  Retaining PACE's role split here
  # still constructs a two-pool rollout storage and makes a one-environment viewer
  # fail before it can load that actor.  The environment roles have already been
  # removed by ``_configure_rollout_motion``; mirror that inference-only state in
  # the runner so Stage-II checkpoints can be played as a single robot.
  algorithm_cfg["acquisition_fraction"] = None
  algorithm_cfg["consolidation_enabled"] = False
  algorithm_cfg["use_star"] = False
  algorithm_cfg["base_checkpoint"] = None
  return runner_cfg


def build_env_and_policy(
  task_id: str,
  checkpoint: str,
  num_envs: int,
  device: str,
  play: bool | None = None,
  manifest: str | None = None,
  eval_mode: Literal["nominal", "randomized"] | None = None,
  random_recovery_start: bool = False,
):
  """Returns ``(env, policy, command)``.

  Args:
    play: Compatibility switch. True maps to ``nominal`` and False maps to
      ``randomized``.
    eval_mode: Explicit rollout mode. ``nominal`` is clean measurement;
      ``randomized`` retains training perturbations for stratification (Sec. IV-C).
    manifest: Override the clip manifest baked into the registered task.
    random_recovery_start: Make every reset sample a randomized unstable pose and
      disable the upward assistance force. Intended for interactive visualization.
  """
  configure_torch_backends()

  mode = _resolve_eval_mode(play=play, eval_mode=eval_mode)
  # Random recovery needs the construction-time assistance event even though the
  # visualization zeros its force. Loading the training-shaped config guarantees the
  # event exists; the helper below strips all other perturbations in nominal mode.
  env_cfg = load_env_cfg(task_id, play=mode == "nominal" and not random_recovery_start)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = num_envs

  _configure_rollout_motion(env_cfg, manifest)
  if random_recovery_start:
    _configure_random_recovery_rollout(env_cfg, nominal=mode == "nominal")
  _configure_rollout_env_cfg(env_cfg)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id)
  assert runner_cls is not None, f"Task {task_id} has no runner class registered."
  runner = runner_cls(
    wrapped, _inference_runner_cfg(agent_cfg), log_dir=None, device=device
  )
  # This harness replaces the training library with evaluation subsets or
  # post-Stage-I logical clips. Restore policy weights, but never restore the
  # checkpoint's sampler state: its clip IDs/max_bins describe the original
  # training library and are intentionally incompatible with this rollout library.
  runner.load(
    checkpoint,
    load_cfg={
      "actor": True,
      "critic": False,
      "optimizer": False,
      "iteration": False,
      "rnd": False,
    },
    restore_env_state=False,
  )
  policy = runner.get_inference_policy(device=device)

  command = cast(MultiMotionCommand, env.command_manager.get_term("motion"))
  return wrapped, policy, command


def resolve_device(device: str | None) -> str:
  if device is not None:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"
