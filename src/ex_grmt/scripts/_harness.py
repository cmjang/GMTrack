"""Build a trained-policy environment outside mjlab's train/play CLIs.

`stratify.py` and `evaluate.py` both need "env + loaded policy, no viewer, no logging".
mjlab's `play.py` does this inline, tangled with W&B artifact resolution and viewer
setup, so the few lines that matter are lifted here.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from ex_grmt.mdp.commands import MultiMotionCommand


def build_env_and_policy(
  task_id: str,
  checkpoint: str,
  num_envs: int,
  device: str,
  play: bool,
  manifest: str | None = None,
):
  """Returns ``(env, policy, command)``.

  Args:
    play: True for clean measurement (no observation corruption, no pushes); False to
      keep domain randomization active, which is what "randomized rollouts" means for
      stratification (Sec. IV-C).
    manifest: Override the clip manifest baked into the registered task.
  """
  configure_torch_backends()

  env_cfg = load_env_cfg(task_id, play=play)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = num_envs

  motion_cfg = env_cfg.commands["motion"]
  if manifest is not None:
    motion_cfg.manifest = manifest
  # Evaluation drives clips explicitly via set_clip; the training sampler must not
  # reassign them underneath us on reset.
  motion_cfg.sampling_mode = "start"
  # A shorter episode limit would time out long clips before they finish.
  env_cfg.episode_length_s = int(1e9)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id)
  assert runner_cls is not None, f"Task {task_id} has no runner class registered."
  runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(checkpoint)
  policy = runner.get_inference_policy(device=device)

  command = cast(MultiMotionCommand, env.command_manager.get_term("motion"))
  return wrapped, policy, command


def resolve_device(device: str | None) -> str:
  if device is not None:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"
