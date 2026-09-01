"""Launch a Viser web viewer for a local GMTrack checkpoint and manifest.

Unlike mjlab's generic ``play`` command, this entry point uses the evaluation
harness.  It therefore loads only policy weights and never restores the training
motion sampler into a deliberately smaller visualization manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mjlab
import torch
import tyro
import viser
from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.viewer import ViserPlayViewer

from gmtrack.scripts._harness import build_env_and_policy, resolve_device

_PUSH_DIRECTIONS = ("+X", "-X", "+Y", "-Y")


def _manual_push_velocity_range(
  direction: str, speed_m_s: float
) -> dict[str, tuple[float, float]]:
  """Build a deterministic horizontal velocity impulse for the viewer."""
  if speed_m_s <= 0.0:
    raise ValueError("Manual push speed must be positive.")
  vectors = {
    "+X": ("x", speed_m_s),
    "-X": ("x", -speed_m_s),
    "+Y": ("y", speed_m_s),
    "-Y": ("y", -speed_m_s),
  }
  try:
    axis, value = vectors[direction]
  except KeyError as exc:
    raise ValueError(f"Unknown manual push direction: {direction!r}") from exc
  return {axis: (value, value)}


class GMTrackWebplayViewer(ViserPlayViewer):
  """Viser viewer with a selected-environment horizontal push control."""

  def setup(self) -> None:
    super().setup()
    with self._server.gui.add_folder("Recovery Test"):
      self._push_speed = self._server.gui.add_slider(
        "Push speed (m/s)", min=0.1, max=3.0, step=0.1, initial_value=1.0
      )
      push_buttons = self._server.gui.add_button_group(
        "Push selected robot", options=_PUSH_DIRECTIONS
      )

      @push_buttons.on_click
      def _(event) -> None:
        self.request_action(
          "CUSTOM",
          {
            "type": "manual_push",
            "direction": event.target.value,
            "speed_m_s": float(self._push_speed.value),
          },
        )

  def _handle_custom_action(self, action, payload) -> bool:
    if isinstance(payload, dict) and payload.get("type") == "manual_push":
      env = self.env.unwrapped
      env_ids = torch.tensor([self._scene.env_idx], device=env.device)
      velocity_range = _manual_push_velocity_range(
        payload["direction"], payload["speed_m_s"]
      )
      with self._sim_lock:
        push_by_setting_velocity(
          env,
          env_ids,
          velocity_range,
          SceneEntityCfg("robot"),
        )
        env.sim.forward()
        env.sim.sense()
      return True
    return super()._handle_custom_action(action, payload)


@dataclass
class Config:
  checkpoint: str
  manifest: str = (
    "data/current/"
    "stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"
  )
  task: str = "GMTrack-Stage1-Flat-Unitree-G1"
  num_envs: int = 1
  device: str | None = None
  eval_mode: Literal["nominal", "randomized"] = "nominal"
  random_recovery_start: bool = False
  port: int = 8080


def main(cfg: Config) -> None:
  device = resolve_device(cfg.device)
  env, policy, _command = build_env_and_policy(
    task_id=cfg.task,
    checkpoint=cfg.checkpoint,
    num_envs=cfg.num_envs,
    device=device,
    manifest=cfg.manifest,
    eval_mode=cfg.eval_mode,
    random_recovery_start=cfg.random_recovery_start,
  )
  server = viser.ViserServer(port=cfg.port, label="GMTrack")
  try:
    GMTrackWebplayViewer(env, policy, viser_server=server).run()
  finally:
    env.close()
    server.stop()


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
