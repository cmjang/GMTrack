"""On-policy runner for Extreme-RGMT.

Differences from mjlab's ``MotionTrackingOnPolicyRunner``:

* **No motion data is bundled into the ONNX file.** mjlab's tracking runner packs the
  whole reference clip into the exported model as buffers, which works for
  single-clip replay but is untenable for a multi-hour library -- and is the wrong
  interface anyway. Extreme-RGMT's deployment target is *online teleoperation*, where
  the reference window arrives frame by frame from an inertial capture stream, so the
  exported policy takes the reference window as an ordinary input.
  :class:`~ex_grmt.rsl_rl.models.ExGRMTActor` already exposes exactly the three named
  inputs the on-robot runtime has to fill.
* Extra metadata is attached so the deployment side can reconstruct the observation
  layout without reading this repo.
"""

from __future__ import annotations

import os
from typing import cast

import torch
from mjlab.entity import Entity
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env.vec_env import VecEnv

from ex_grmt.mdp.commands import MultiMotionCommand
from ex_grmt.rsl_rl.models import REQUIRED_GROUPS as ACTOR_OBS_GROUPS


class ExGRMTOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that exports a teleoperation-ready policy and logs PACE/STAR metrics."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    policy = self.alg.get_policy()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[ex-grmt] actor parameters: {n_params / 1e6:.2f}M")

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
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      attach_metadata_to_onnx(str(onnx_path), self._build_metadata())
    except Exception as e:  # noqa: BLE001 - export must never kill a training run
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def _build_metadata(self) -> dict:
    """Deployment metadata for the exported policy.

    mjlab's ``get_base_metadata`` cannot be reused: it hard-codes a single observation
    group literally named ``"actor"``, whereas this policy takes three named inputs
    (proprioceptive history, action history, reference window). The robot-level fields
    below mirror it so downstream tooling keeps working; the observation section
    describes our actual export interface instead.
    """
    env = self.env.unwrapped
    robot: Entity = env.scene["robot"]
    joint_action = env.action_manager.get_term("joint_pos")

    joint_name_to_ctrl_id = {
      a.target.split("/")[-1]: a.id for a in robot.spec.actuators
    }
    ctrl_ids = [
      joint_name_to_ctrl_id[j]
      for j in robot.joint_names
      if j in joint_name_to_ctrl_id
    ]
    stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
    damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]

    motion = cast(MultiMotionCommand, env.command_manager.get_term("motion"))
    actor_cfg = self.cfg.get("actor", {})
    scale = joint_action._scale
    obs_mgr = env.observation_manager

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
      # Export interface: three named inputs, in this order.
      "policy_input_names": list(ACTOR_OBS_GROUPS),
      "history_length": actor_cfg.get("history_length"),
      "proprio_term_names": list(obs_mgr.active_terms["proprio_hist"]),
      "proprio_term_dims": list(actor_cfg.get("proprio_term_dims", ())),
      # Reference window layout: [v_ref(3), w_ref(3), g_ref(3), q_ref(J)] per token,
      # each token egocentric in the reference root frame at its own timestamp.
      "command_window_radius": motion.cfg.command_window_radius,
      "command_window_tokens": motion.num_window_tokens,
      "command_token_dim": motion.command_token_dim,
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
