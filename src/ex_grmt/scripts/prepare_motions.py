"""Convert retargeted motion CSVs into the packed ``.npz`` clips training consumes.

Pipeline: CSV (Unitree generalized coordinates) -> 10-second slices -> forward
kinematics replay in MuJoCo -> ``.npz`` + manifest entry.

Why not just call ``mjlab.scripts.csv_to_npz``: that script hard-codes a
Weights & Biases artifact upload and always writes to ``/tmp/motion.npz``, which does
not survive batching thousands of clips. This module reuses mjlab's CSV
``MotionLoader`` -- where the load-bearing correctness lives (xyzw -> wxyz quaternion
conversion, fps resampling, SO(3) finite-difference velocities) -- and replaces only
the I/O around it.

CRITICAL: the body ordering in the output must come from MuJoCo's depth-first body
traversal. Converters written for IsaacLab produce a breadth-first ordering and the
resulting files silently track the wrong links. Always generate clips with this
script (or mjlab's), never with an IsaacLab-derived one.

Usage::

    uv run python -m ex_grmt.scripts.prepare_motions \\
        --input-dir data/raw/lafan1 --source lafan1 --input-fps 30

    uv run python -m ex_grmt.scripts.prepare_motions \\
        --input-dir data/raw/seed --source seed --input-fps 30 --append
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab
import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from tqdm import tqdm

# Column order of the CSV joint block, mirroring mjlab.scripts.csv_to_npz.main.
# This is the Unitree G1 29-DoF convention; do not reorder.
G1_JOINT_ORDER: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

_LOG_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


@dataclass
class Config:
  input_dir: str
  """Directory of retargeted CSVs (searched recursively for ``*.csv``)."""
  source: str
  """Provenance tag recorded in the manifest, e.g. ``lafan1`` or ``seed``."""
  input_fps: float = 30.0
  """Frame rate of the CSVs. LAFAN1 retargeted data is 30 fps."""
  output_fps: float = 50.0
  """Control rate. Must match the policy rate (decimation 4 x 0.005 s)."""
  clip_seconds: float = 10.0
  """Sequences longer than this are split; shorter ones are kept whole (Sec. IV-C).

  Must not exceed the environment's ``episode_length_s`` or clips become
  uncompletable and stratification scores them all as failures.
  """
  min_clip_seconds: float = 1.0
  """Trailing remainders shorter than this are dropped rather than kept as stubs."""
  output_dir: str = "data/motions"
  manifest: str = "data/manifests/all.json"
  append: bool = False
  """Merge into an existing manifest instead of overwriting it."""
  device: str = "cuda:0"
  limit: int | None = None
  """Process at most this many CSVs. Useful for smoke tests."""
  overwrite: bool = False
  """Re-convert clips whose ``.npz`` already exists."""


def _replay(
  sim: Simulation,
  scene: Scene,
  robot_joint_indexes: list[int],
  motion: MotionLoader,
) -> dict[str, np.ndarray]:
  """Drive the reference through the simulator and read back body kinematics.

  Positions and velocities are *written* to the model and only forward kinematics is
  evaluated, so this is a pure coordinate transform, not a physics rollout.
  """
  robot: Entity = scene["robot"]
  n = motion.output_frames
  device = robot.data.joint_pos.device
  n_joints = robot.data.joint_pos.shape[1]
  n_bodies = robot.data.body_link_pos_w.shape[1]

  # Accumulate on-device and transfer once. Reading each field back per frame costs
  # six host syncs per step, which dominates the runtime of an otherwise trivial
  # forward-kinematics pass (~880 clips x 500 frames for LAFAN1).
  out = {
    "joint_pos": torch.empty(n, n_joints, device=device),
    "joint_vel": torch.empty(n, n_joints, device=device),
    "body_pos_w": torch.empty(n, n_bodies, 3, device=device),
    "body_quat_w": torch.empty(n, n_bodies, 4, device=device),
    "body_lin_vel_w": torch.empty(n, n_bodies, 3, device=device),
    "body_ang_vel_w": torch.empty(n, n_bodies, 3, device=device),
  }
  ref_lin_vel = torch.empty(n, 3, device=device)
  ref_ang_vel = torch.empty(n, 3, device=device)

  scene.reset()
  for i in range(n):
    (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = (
      motion.get_next_state()
    )

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    out["joint_pos"][i] = robot.data.joint_pos[0]
    out["joint_vel"][i] = robot.data.joint_vel[0]
    out["body_pos_w"][i] = robot.data.body_link_pos_w[0]
    out["body_quat_w"][i] = robot.data.body_link_quat_w[0]
    out["body_lin_vel_w"][i] = robot.data.body_link_lin_vel_w[0]
    out["body_ang_vel_w"][i] = robot.data.body_link_ang_vel_w[0]
    ref_lin_vel[i] = base_lin_vel[0]
    ref_ang_vel[i] = base_ang_vel[0]

  # Round-trip check: what we wrote to the free joint must come back out of forward
  # kinematics. This guards the failure mode that matters -- a wrong quaternion
  # convention or joint ordering -- which shows up as an O(0.1-10) discrepancy.
  #
  # Tolerance is loosened from torch's float32 default (atol 1e-5, rtol 1.3e-6),
  # which is tighter than the write -> mj_forward -> read path can hold: LAFAN1
  # trips it at ~1.4e-5 m/s on fast frames. mjlab's own csv_to_npz asserts at the
  # default and crashes on the same data. 1e-3 still catches every convention error
  # by three orders of magnitude.
  torch.testing.assert_close(
    out["body_lin_vel_w"][:, 0], ref_lin_vel, atol=1e-3, rtol=1e-3
  )
  torch.testing.assert_close(
    out["body_ang_vel_w"][:, 0], ref_ang_vel, atol=1e-3, rtol=1e-3
  )

  return {k: v.cpu().numpy() for k, v in out.items()}


def _slice_ranges(
  num_rows: int, rows_per_clip: int, min_rows: int
) -> list[tuple[int, int]]:
  """1-indexed inclusive ``line_range`` pairs covering the CSV.

  Paper Sec. IV-C: "partition each motion sequence longer than 10 s into 10-s clips
  while retaining shorter sequences as individual clips."

  No clip may exceed ``rows_per_clip``. An earlier version folded a short trailing
  remainder into the previous slice, which produced clips of up to 12.5 s -- longer
  than ``episode_length_s``, so they could never be tracked to completion and
  stratification would score every one of them as a failure.

  A trailing remainder shorter than ``min_rows`` is dropped instead: it is too short
  to carry a meaningful completion rate, and a 0.2 s "clip" would just add noise to
  the mastered/challenging split. A whole source sequence shorter than ``min_rows``
  is still kept, per the sentence above.
  """
  if num_rows <= rows_per_clip:
    return [(1, num_rows)]

  ranges: list[tuple[int, int]] = []
  start = 1
  while start <= num_rows:
    stop = min(start + rows_per_clip - 1, num_rows)
    length = stop - start + 1
    if length >= min_rows or not ranges:
      ranges.append((start, stop))
    start = stop + 1
  return ranges


def main(cfg: Config) -> None:
  if cfg.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA requested but unavailable. Pass --device cpu explicitly if that is "
      "really what you want (it is roughly 50x slower)."
    )

  input_dir = Path(cfg.input_dir)
  csv_files = sorted(input_dir.rglob("*.csv"))
  if not csv_files:
    raise FileNotFoundError(f"No .csv files under {input_dir.resolve()}")
  if cfg.limit is not None:
    csv_files = csv_files[: cfg.limit]

  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = Path(cfg.manifest)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / cfg.output_fps
  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=cfg.device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=cfg.device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(list(G1_JOINT_ORDER), preserve_order=True)[0]

  clips: list[dict[str, Any]] = []
  if cfg.append and manifest_path.exists():
    with manifest_path.open() as f:
      clips = json.load(f)["clips"]
    existing = {c["name"] for c in clips}
  else:
    existing = set()

  rows_per_clip = int(round(cfg.clip_seconds * cfg.input_fps))
  min_rows = int(round(cfg.min_clip_seconds * cfg.input_fps))

  for csv_path in tqdm(csv_files, desc="clips", unit="file"):
    num_rows = sum(1 for _ in csv_path.open())
    ranges = _slice_ranges(num_rows, rows_per_clip, min_rows)
    multi = len(ranges) > 1

    for i, line_range in enumerate(ranges):
      name = f"{cfg.source}__{csv_path.stem}"
      if multi:
        name = f"{name}__{i:03d}"
      out_path = output_dir / f"{name}.npz"

      if out_path.exists() and not cfg.overwrite:
        if name not in existing:
          data = np.load(out_path)
          clips.append(
            _entry(name, cfg, out_path, manifest_path, data["joint_pos"].shape[0])
          )
          existing.add(name)
        continue

      motion = MotionLoader(
        motion_file=str(csv_path),
        input_fps=int(cfg.input_fps),
        output_fps=int(cfg.output_fps),
        device=cfg.device,
        line_range=line_range,
      )
      log = _replay(sim, scene, robot_joint_indexes, motion)
      np.savez(out_path, fps=np.array([cfg.output_fps]), **log)

      if name in existing:
        clips = [c for c in clips if c["name"] != name]
      clips.append(
        _entry(name, cfg, out_path, manifest_path, log["joint_pos"].shape[0])
      )
      existing.add(name)

  clips.sort(key=lambda c: c["name"])
  total_frames = sum(c["num_frames"] for c in clips)
  with manifest_path.open("w") as f:
    json.dump({"clips": clips}, f, indent=2)

  print(
    f"[ex-grmt] wrote {len(clips)} clips "
    f"({total_frames / cfg.output_fps / 60:.1f} min) to {manifest_path}"
  )


def _entry(
  name: str, cfg: Config, out_path: Path, manifest_path: Path, num_frames: int
) -> dict[str, Any]:
  return {
    "name": name,
    "source": cfg.source,
    # Stored relative to the manifest so the whole data directory can be rsync'd to
    # the cluster without rewriting paths. MotionLibrary.from_manifest resolves it.
    "path": os.path.relpath(out_path.resolve(), manifest_path.parent.resolve()),
    "num_frames": int(num_frames),
    "fps": float(cfg.output_fps),
  }


if __name__ == "__main__":
  main(tyro.cli(Config, config=mjlab.TYRO_FLAGS))
