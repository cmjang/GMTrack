"""On-policy runner for GMTrack.

Differences from mjlab's ``MotionTrackingOnPolicyRunner``:

* **No motion data is bundled into the ONNX file.** mjlab's tracking runner packs the
  whole reference clip into the exported model as buffers, which works for
  single-clip replay but is untenable for a multi-hour library -- and is the wrong
  interface anyway. GMTrack's deployment target is *online teleoperation*, where
  the reference window arrives frame by frame from an inertial capture stream, so the
  exported policy takes the reference window as an ordinary input.
  :class:`~gmtrack.rsl_rl.models.GMTrackActor` exposes the named inputs the on-robot
  runtime has to fill, including a validity mask for causal-window startup padding.
* Extra metadata is attached so the deployment side can reconstruct the observation
  layout without reading this repo.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal, cast

import torch
from mjlab.entity import Entity
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env.vec_env import VecEnv

from gmtrack.mdp.commands import MultiMotionCommand
from gmtrack.provenance import build_run_provenance, sha256_file, write_run_provenance
from gmtrack.rsl_rl.storage import TRACKING_FAILURES_EXTRA


def _command_token_layout(command_token_dim: int, joint_ref_dim: int) -> list[str]:
  """Describe the supported reference-token layouts for ONNX consumers."""
  base_dim = 9 + joint_ref_dim
  layout = ["v_ref[3]", "w_ref[3]", "g_ref[3]", f"q_ref[{joint_ref_dim}]"]
  if command_token_dim == base_dim:
    return layout
  if command_token_dim == base_dim + 6:
    return [*layout, "root_ori_error[6]"]
  raise ValueError(
    f"Cannot describe command_token_dim={command_token_dim}: expected {base_dim} "
    f"(open-loop) or {base_dim + 6} (heading closed-loop)."
  )


class GMTrackOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that exports a teleoperation-ready policy and logs PACE/STAR metrics."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    provenance_cfg = copy.deepcopy(train_cfg)
    super().__init__(env, train_cfg, log_dir, device)
    self._install_tracking_failure_step_hook()
    if log_dir is not None:
      self._validate_stage2_base_checkpoint(provenance_cfg)
    policy = self.alg.get_policy()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[gmtrack] actor parameters: {n_params / 1e6:.2f}M")
    if log_dir is not None and self.gpu_global_rank == 0:
      self._write_training_provenance(log_dir, provenance_cfg)

  def _validate_stage2_base_checkpoint(self, train_cfg: dict) -> None:
    """Require Stage II to warm-start from the checkpoint that produced D_m/D_c."""
    command = self._motion_command()
    if command is None or not command.cfg.require_v1_stratification:
      return
    report = command.stratification_report
    if not isinstance(report, dict):
      raise ValueError(
        "Strict Stage-II command has no validated stratification report."
      )
    base_checkpoint = train_cfg["algorithm"].get("base_checkpoint")
    if not isinstance(base_checkpoint, str) or not base_checkpoint:
      raise ValueError(
        "Strict v1 Stage II requires the Stage-I base checkpoint used for stratification."
      )
    expected = report["provenance"]["base_checkpoint_sha256"]
    actual = sha256_file(base_checkpoint)
    if actual != expected:
      raise ValueError(
        "Stage-II base checkpoint does not match the checkpoint that produced "
        f"D_m/D_c: expected SHA256 {expected}, got {actual}."
      )

  def _install_tracking_failure_step_hook(self) -> None:
    """Expose Gymnasium ``terminated`` separately from the wrapper's combined done.

    mjlab's rsl-rl wrapper returns only ``terminated | truncated`` plus timeout
    metadata. PACE Eq. 17 needs the true failure signal: deriving it from those two
    values loses a failure when termination and timeout happen on the same step.
    """
    original_step = self.env.step

    def step_with_tracking_failures(actions: torch.Tensor):
      obs, rewards, dones, extras = original_step(actions)
      terminated = self.env.unwrapped.termination_manager.terminated
      if not isinstance(terminated, torch.Tensor):
        raise TypeError("termination_manager.terminated must be a torch.Tensor.")
      extras[TRACKING_FAILURES_EXTRA] = terminated.detach().clone()
      return obs, rewards, dones, extras

    self.env.step = step_with_tracking_failures  # type: ignore[method-assign]

  def _write_training_provenance(self, log_dir: str, train_cfg: dict) -> None:
    """Fail closed if the exact code/data inputs of a training run cannot be saved."""
    command = self._motion_command()
    if command is None:
      raise ValueError("GMTrack training provenance requires a motion command.")
    cfg = command.cfg
    algorithm_cfg = train_cfg.get("algorithm", {})
    payload = build_run_provenance(
      repo_root=Path(__file__).resolve().parents[3],
      train_cfg=train_cfg,
      manifests={
        "motion_library": cfg.manifest,
        "challenging": cfg.acquisition_clips
        if isinstance(cfg.acquisition_clips, str)
        else None,
        "mastered": cfg.consolidation_clips
        if isinstance(cfg.consolidation_clips, str)
        else None,
        "validated_challenging": cfg.stratification_challenging_manifest,
        "validated_mastered": cfg.stratification_mastered_manifest,
      },
      base_checkpoint=algorithm_cfg.get("base_checkpoint"),
      recovery_probability=float(cfg.recovery_probability),
      observation_schema=self.alg.observation_schema,
    )
    write_run_provenance(Path(log_dir) / "run_provenance.json", payload)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,
      output_names=onnx_model.output_names,
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None) -> None:
    infos = {**(infos or {}), "gmtrack_env_state": self._collect_env_state()}
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      attach_metadata_to_onnx(str(onnx_path), self._build_metadata())
    except Exception as e:  # noqa: BLE001 - export must never kill a training run
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
    optimizer_lr: Literal["checkpoint", "config"] | None = None,
    restore_env_state: bool = True,
  ) -> dict:
    """Restore the model plus GMTrack's adaptive environment state.

    Args:
      optimizer_lr: ``"checkpoint"`` gives an exact training resume. ``"config"``
        restores optimizer moments but intentionally reapplies the current runner
        config's learning rate. When omitted, use the runner config's
        ``resume_optimizer_lr`` setting (default: ``"checkpoint"``).
      restore_env_state: Restore adaptive samplers and the recovery anneal clock for
        an exact training resume. Evaluation and stratification must set this false:
        they replace the training motion library with a subset or logical clips, so
        the saved sampler shape and clip IDs deliberately do not match.
    """
    if optimizer_lr is None:
      optimizer_lr = self.cfg.get("resume_optimizer_lr", "checkpoint")
    if optimizer_lr not in ("checkpoint", "config"):
      raise ValueError(
        f"optimizer_lr must be 'checkpoint' or 'config', got {optimizer_lr!r}."
      )
    if load_cfg is None:
      effective_load_cfg: dict[str, Any] = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
        "rnd": True,
        "optimizer_lr": optimizer_lr,
      }
    else:
      effective_load_cfg = dict(load_cfg)
      effective_load_cfg["optimizer_lr"] = optimizer_lr

    infos = super().load(path, effective_load_cfg, strict, map_location)
    if restore_env_state and infos and "gmtrack_env_state" in infos:
      self._restore_env_state(infos["gmtrack_env_state"])
    return infos

  def _motion_command(self) -> MultiMotionCommand | None:
    """Return the motion term when this runner owns a GMTrack environment."""
    env = self.env.unwrapped
    manager = getattr(env, "command_manager", None)
    if manager is None:
      return None
    try:
      command = manager.get_term("motion")
    except (KeyError, ValueError):
      return None
    return command if isinstance(command, MultiMotionCommand) else None

  def _collect_env_state(self) -> dict[str, Any]:
    """Capture persistent adaptive state omitted by mjlab.

    Episode-local recovery masks/forces are deliberately not checkpointed: mjlab
    does not serialize MuJoCo physics state, so restoring those flags onto a freshly
    reset simulation would create an inconsistent episode. The recovery-assistance
    anneal *clock* is checkpointed here, because it must count only steps taken with
    fall recovery enabled -- mjlab's ``common_step_counter`` covers the whole training
    history and would report an already-exhausted anneal the moment recovery is
    switched on partway through a run.
    """
    state: dict[str, Any] = {"version": 2}
    command = self._motion_command()
    if command is None:
      return state

    state["recovery_steps_elapsed"] = int(command.recovery_steps_elapsed)

    samplers: dict[str, dict[str, Any]] = {}
    for name in ("sampler_acq", "sampler_con"):
      sampler = getattr(command, name, None)
      if sampler is None:
        continue
      samplers[name] = {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in sampler.state_dict().items()
      }
    state["samplers"] = samplers
    return state

  def _restore_env_state(self, state: dict[str, Any]) -> None:
    """Restore adaptive state, rejecting incompatible data/env layouts."""
    version = state.get("version")
    if version not in (1, 2):
      raise ValueError(f"Unsupported GMTrack environment state version {version!r}.")
    command = self._motion_command()
    if command is None:
      return

    # The env owns a motion command, so the checkpoint must describe one. A state
    # without ``samplers`` was written by a runner whose env had no motion term;
    # defaulting it to empty would silently resume with a cold adaptive sampler.
    if "samplers" not in state:
      raise ValueError(
        "GMTrack environment state has no 'samplers' entry, but this environment has "
        "a motion command. The checkpoint was written for a different env layout."
      )
    saved_samplers = state["samplers"]
    if not isinstance(saved_samplers, dict):
      raise ValueError("GMTrack environment state 'samplers' entry must be a mapping.")
    expected_names = {
      name
      for name in ("sampler_acq", "sampler_con")
      if getattr(command, name, None) is not None
    }
    saved_names = set(saved_samplers)
    if saved_names != expected_names:
      raise ValueError(
        "Checkpoint sampler layout does not match the current motion manifest: "
        f"expected {sorted(expected_names)}, found {sorted(saved_names)}."
      )
    for name, sampler_state in saved_samplers.items():
      sampler = getattr(command, name)
      sampler.load_state_dict(sampler_state, strict=True)

    # Version 1 predates the recovery-local anneal clock. Such a checkpoint carries
    # no anneal progress to restore -- either recovery was off, or it ran against
    # ``common_step_counter`` and the assistance force was already zero throughout.
    # Zero is therefore the only defined value, not a guess papering over a gap.
    if version == 1:
      command.recovery_steps_elapsed = 0
      print(
        "[gmtrack] checkpoint predates the recovery-local assist anneal; "
        "restarting the anneal clock at 0."
      )
    elif "recovery_steps_elapsed" not in state:
      raise ValueError(
        "Version-2 GMTrack environment state is missing 'recovery_steps_elapsed'."
      )
    else:
      command.recovery_steps_elapsed = int(state["recovery_steps_elapsed"])

  def _build_metadata(self) -> dict:
    """Deployment metadata for the exported policy.

    mjlab's ``get_base_metadata`` cannot be reused: it hard-codes a single observation
    group literally named ``"actor"``, whereas this policy takes separate named
    history/reference inputs and causal policies additionally take validity masks.
    The robot-level fields below mirror it so downstream tooling keeps working; the
    observation section describes our actual export interface instead.
    """
    env = self.env.unwrapped
    robot: Entity = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")

    joint_name_to_ctrl_id = {
      a.target.split("/")[-1]: a.id for a in robot.spec.actuators
    }
    ctrl_ids = [
      joint_name_to_ctrl_id[j] for j in robot.joint_names if j in joint_name_to_ctrl_id
    ]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]

    motion = cast(MultiMotionCommand, env.command_manager.get_term("motion"))
    actor_cfg = self.cfg["actor"]
    actor_command_token_dim = int(actor_cfg["command_token_dim"])
    if actor_command_token_dim != motion.command_token_dim:
      raise ValueError(
        "Actor/environment command-token mismatch: actor command_token_dim="
        f"{actor_command_token_dim}, environment command_token_dim="
        f"{motion.command_token_dim}."
      )
    proprio_term_dims = list(actor_cfg["proprio_term_dims"])
    if not proprio_term_dims:
      raise ValueError("Actor metadata requires non-empty proprio_term_dims.")
    command_layout = _command_token_layout(
      actor_command_token_dim, int(proprio_term_dims[-1])
    )
    heading_closed_loop = command_layout[-1] == "root_ori_error[6]"
    scale = joint_action._scale
    obs_mgr = env.observation_manager
    observation_schema = self.alg.observation_schema
    schema = observation_schema["schema"]
    command_schema = schema["command"]
    history_mask_schema = schema["mask_layout"]["history_valid_mask"]
    past_mask_schema = schema["mask_layout"]["past_valid_mask"]

    metadata: dict = {
      "run_path": self._run_name(),
      "joint_names": list(robot.joint_names),
      "joint_stiffness": stiffness.tolist(),
      "joint_damping": damping.tolist(),
      "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
      "action_scale": scale[0].cpu().tolist()
      if isinstance(scale, torch.Tensor)
      else scale,
      "command_names": list(env.command_manager.active_terms),
      "policy_input_names": schema["actor_observation_groups"],
      "observation_schema_sha256": observation_schema["sha256"],
      "history_length": actor_cfg["history_length"],
      "proprio_term_names": list(obs_mgr.active_terms["proprio_hist"]),
      "proprio_term_dims": proprio_term_dims,
      # Each token is egocentric in the reference root frame at its own timestamp.
      # Heading policies append the current root-orientation error and therefore
      # require a one-time deployment-frame alignment at reset.
      "command_window_radius": motion.cfg.command_window_radius,
      "command_window_offsets": motion.window_offsets.detach().cpu().tolist(),
      "command_window_offsets_seconds": [
        offset / command_schema["fps"] for offset in command_schema["window_offsets"]
      ],
      "command_window_tokens": motion.num_window_tokens,
      "command_window_fps": command_schema["fps"],
      "command_window_offset_unit": command_schema["offset_unit"],
      "command_window_ordering": command_schema["ordering"],
      "command_window_startup_fill": command_schema["startup_fill"],
      "command_window_position_encoding": command_schema["position_encoding"],
      "command_window_position_normalization": command_schema["position_normalization"],
      "command_window_valid_mask_group": (
        past_mask_schema["observation_group"]
        if past_mask_schema is not None
        else "none"
      ),
      "command_window_valid_mask_rule": (
        f"true: {past_mask_schema['true']}; false: {past_mask_schema['false']}"
        if past_mask_schema is not None
        else "none"
      ),
      "history_valid_mask_group": (
        history_mask_schema["observation_group"]
        if history_mask_schema is not None
        else "none"
      ),
      "history_valid_mask_rule": (
        f"true: {history_mask_schema['true']}; false: {history_mask_schema['false']}"
        if history_mask_schema is not None
        else "none"
      ),
      "intent_auxiliary_exported": False,
      "intent_posterior_mean_exported": bool(
        schema["intent_auxiliary"].get("actor_conditioning")
        == "deterministic_posterior_mean"
      ),
      "command_token_dim": actor_command_token_dim,
      "command_token_layout": command_layout,
      "heading_closed_loop": heading_closed_loop,
      "heading_body_name": motion.cfg.body_names[0] if heading_closed_loop else "none",
      "heading_error_convention": (
        "matrix_from_quat(inv(q_robot_heading_body_current) * "
        "q_ref_heading_body_token)[..., :2].reshape(row-major)"
        if heading_closed_loop
        else "none"
      ),
      "initial_yaw_alignment_contract": (
        "At deployment reset compute q_align = yaw(q_robot_initial) * "
        "inv(yaw(q_ref_initial)); keep q_align fixed for the rollout, set "
        "q_ref_aligned = q_align * q_ref_token, and recompute root_ori_error from "
        "q_robot_current and q_ref_aligned every policy step."
        if heading_closed_loop
        else "none"
      ),
      "anchor_body_name": motion.cfg.anchor_body_name,
      "tracked_body_names": list(motion.cfg.body_names),
    }
    return metadata

  def _run_name(self) -> str:
    if self.logger.logger_type in ("wandb", "WandbLogWriter"):
      import wandb

      if wandb.run is not None:
        return wandb.run.name
    return "local"
