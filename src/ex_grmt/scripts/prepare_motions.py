"""Convert retargeted motion CSVs into complete ``.npz`` sequences for Stage I.

Pipeline: CSV (Unitree generalized coordinates) -> forward-kinematics replay in
MuJoCo -> one ``.npz`` + manifest entry per source sequence. Paper Sec. IV-C applies
the 10-second split only *after* Stage-I training, when :mod:`stratify` evaluates the
base policy and constructs the mastered/challenging sets.

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
        --input-dir data/datasets/raw/lafan1 --source lafan1 --input-format mjlab --input-fps 30 \\
        --output-dir data/datasets/stage1_full/lafan1 \\
        --manifest logs/data_build/manifests/lafan1_full.json

    uv run python -m ex_grmt.scripts.prepare_motions \\
        --input-dir data/datasets/raw/seed_backflip --source seed-backflip --input-fps 120 \
        --input-format bones-seed --output-dir data/datasets/seed_backflip \
        --manifest logs/data_build/manifests/seed_backflip.json

"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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

from ex_grmt.assets import get_ex_grmt_g1_robot_cfg
from ex_grmt.motion_grounding import (
  DEFAULT_CLEARANCE,
  CorrectionSmoothing,
  G1MotionGrounder,
  GroundClearanceReport,
  GroundingResult,
)

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

InputFormat = Literal["mjlab", "bones-seed", "motiondecode"]
GroundAlignment = Literal["none", "g1_collision"]

MOTIONDECODE_HEADER: tuple[str, ...] = (
  "root_pos_x(m)",
  "root_pos_y(m)",
  "root_pos_z(m)",
  "root_rot_w",
  "root_rot_x",
  "root_rot_y",
  "root_rot_z",
  *(f"dof_{joint}(rad)" for joint in G1_JOINT_ORDER),
)

_LOG_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)

_NON_MOTION_CSV_NAMES = frozenset({"filtered_metadata.csv"})
_CONVERTER_SCHEMA_VERSION = 4
_GROUND_QC_KEYS = (
  "ground_alignment",
  "ground_clearance_m",
  "ground_smoothing_radius_s",
  "ground_min_clearance_before_m",
  "ground_min_clearance_after_m",
  "ground_max_correction_m",
  "ground_affected_frame_ratio",
  "ground_correction_vmax_mps",
  "ground_correction_amax_mps2",
)


@dataclass
class Config:
  input_dir: str
  """Directory of retargeted CSVs (searched recursively for ``*.csv``)."""
  source: str
  """Provenance tag recorded in the manifest, e.g. ``lafan1`` or ``seed``."""
  input_format: InputFormat
  """Numeric CSV layout and units.

  ``mjlab`` expects a headerless 36-column array in metres/radians with an xyzw
  root quaternion. ``bones-seed`` is the locally preprocessed 36-column BONES-SEED
  layout (root xyz in centimetres, root quaternion in xyzw, then 29 joints in
  degrees). BONES-SEED is recorded at 120 fps. ``motiondecode`` expects the released
  ChingMu/MotionDecode header, metres/radians, and a wxyz root quaternion.
  """
  input_fps: float = 30.0
  """Frame rate of the CSVs. LAFAN1 retargeted data is 30 fps."""
  output_fps: float = 50.0
  """Control rate. Must match the policy rate (decimation 4 x 0.005 s)."""
  replay_batch_frames: int = 500
  """Number of frames replayed in parallel during MuJoCo forward kinematics.

  This only controls conversion memory/throughput; it never slices the saved motion.
  """
  output_dir: str = "data/datasets/stage1_full"
  manifest: str = "logs/data_build/manifests/stage1_lafan_seed_simple.json"
  append: bool = False
  """Merge into an existing manifest instead of overwriting it.

  NOTE: mjlab's tyro configuration disables implicit boolean flags, so this needs an
  explicit value on the command line -- ``--append True``, not a bare ``--append``.
  """
  device: str = "cuda:0"
  limit: int | None = None
  """Process at most this many CSVs. Useful for smoke tests."""
  overwrite: bool = False
  """Re-convert clips whose ``.npz`` already exists."""
  manifest_checkpoint_files: int = 250
  """Atomically checkpoint the manifest after this many source files."""
  selection: str | None = None
  """Optional JSON selection containing ``selected_files`` relative to input_dir."""
  preserve_relative_paths: bool = False
  """Mirror input subdirectories under output_dir and use path-based clip names."""
  skip_invalid_sources: bool = False
  """Record malformed source CSVs and continue instead of aborting the batch."""
  invalid_manifest: str | None = None
  """Optional malformed-source report; defaults beside ``manifest``."""
  ground_alignment: GroundAlignment = "none"
  """Optional root-height alignment against the training G1 collision geometry."""
  ground_clearance_m: float = DEFAULT_CLEARANCE
  """Minimum collision-geometry clearance above the z=0 plane after alignment."""
  ground_smoothing_radius_s: float = 0.3
  """Physical-time radius used to smooth the upward correction envelope."""


def _batch_quat_slerp(
  a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  """Batched equivalent of mjlab's scalar ``quat_slerp`` (wxyz quaternions)."""
  if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4:
    raise ValueError(f"Expected matching (N, 4) quaternions, got {a.shape}, {b.shape}.")
  if blend.shape != (a.shape[0],):
    raise ValueError(f"Expected ({a.shape[0]},) blends, got {blend.shape}.")

  dot = torch.sum(a * b, dim=1)
  eps = torch.finfo(a.dtype).eps * 4.0
  at_start = blend == 0.0
  at_end = blend == 1.0
  degenerate = torch.abs(torch.abs(dot) - 1.0) < eps

  shortest_b = torch.where((dot < 0.0)[:, None], -b, b)
  dot = torch.abs(dot)
  angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
  degenerate |= torch.abs(angle) < eps

  # Rows replaced below still need finite intermediate values.
  sin_angle = torch.sin(angle)
  safe_sin = torch.where(degenerate, torch.ones_like(sin_angle), sin_angle)
  result = (
    a * (torch.sin((1.0 - blend) * angle) / safe_sin)[:, None]
    + shortest_b * (torch.sin(blend * angle) / safe_sin)[:, None]
  )
  result = torch.where(degenerate[:, None], a, result)
  result = torch.where(at_start[:, None], a, result)
  return torch.where(at_end[:, None], b, result)


class _ArrayMotionLoader(MotionLoader):
  """MotionLoader variant that reads an already-loaded source array once."""

  def __init__(
    self,
    motion: np.ndarray,
    input_fps: float,
    output_fps: float,
    device: torch.device | str,
    input_format: InputFormat,
  ) -> None:
    self._motion_array = motion
    self.input_format = input_format
    super().__init__(
      motion_file="<in-memory>",
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
    )

  def _load_motion(self) -> None:
    motion = torch.as_tensor(
      self._motion_array, dtype=torch.float32, device=self.device
    ).clone()
    if motion.ndim != 2 or motion.shape[1] != 36:
      raise ValueError(
        f"Expected a numeric 36-column G1 motion, got shape {tuple(motion.shape)}."
      )
    if self.input_format == "bones-seed":
      # Scale before interpolation so finite-difference velocities are SI too.
      motion[:, :3] *= 0.01
      motion[:, 7:] *= math.pi / 180.0

    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = (
      motion[:, 3:7]
      if self.input_format == "motiondecode"
      else motion[:, [6, 3, 4, 5]]  # xyzw -> wxyz
    )
    self.motion_dof_poss_input = motion[:, 7:]
    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self) -> None:
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    if self.output_frames < 3:
      raise ValueError(
        f"Motion produces only {self.output_frames} output frame(s); need at least 3 "
        "for SO(3) finite-difference velocities."
      )

    # Preserve mjlab's exact float32 frame-index semantics; only the scalar SLERP
    # loop is replaced. Recomputing indices as ``times * input_fps`` drifts at
    # boundary frames and measurably changes angular velocities.
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend[:, None],
    )
    self.motion_base_rots = _batch_quat_slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend[:, None],
    )


def _grounding_qc_arrays(
  cfg: Config, result: GroundingResult | None
) -> dict[str, np.ndarray]:
  """Build scalar, audit-friendly grounding metadata for one output clip."""
  if result is None:
    min_before = math.nan
    min_after = math.nan
    max_correction = 0.0
    affected_ratio = 0.0
    correction_vmax = 0.0
    correction_amax = 0.0
  else:
    correction = np.asarray(result.correction, dtype=np.float64)
    required_correction = np.asarray(result.required_correction, dtype=np.float64)
    min_distance = np.asarray(result.min_distance, dtype=np.float64)
    correction_velocity = np.gradient(correction, 1.0 / cfg.output_fps)
    correction_acceleration = np.gradient(correction_velocity, 1.0 / cfg.output_fps)
    min_before = float(np.min(min_distance))
    # Translating the root and every attached geom upward by dz increases its signed
    # plane distance by exactly dz; no second FK pass is necessary for this scalar.
    min_after = float(np.min(min_distance + correction))
    max_correction = float(np.max(correction))
    # Report frames whose original pose actually needed lifting. Smoothing may
    # deliberately spread a correction into neighbouring clean frames, which is a
    # temporal-filter effect rather than additional source-data penetration.
    affected_ratio = (
      0.0
      if cfg.ground_alignment == "none"
      else float(np.mean(required_correction > 0.0))
    )
    correction_vmax = float(np.max(np.abs(correction_velocity)))
    correction_amax = float(np.max(np.abs(correction_acceleration)))

  return {
    "ground_alignment": np.array([cfg.ground_alignment]),
    "ground_clearance_m": np.array([cfg.ground_clearance_m], dtype=np.float64),
    "ground_smoothing_radius_s": np.array(
      [cfg.ground_smoothing_radius_s], dtype=np.float64
    ),
    "ground_min_clearance_before_m": np.array([min_before], dtype=np.float64),
    "ground_min_clearance_after_m": np.array([min_after], dtype=np.float64),
    "ground_max_correction_m": np.array([max_correction], dtype=np.float64),
    "ground_affected_frame_ratio": np.array([affected_ratio], dtype=np.float64),
    "ground_correction_vmax_mps": np.array([correction_vmax], dtype=np.float64),
    "ground_correction_amax_mps2": np.array([correction_amax], dtype=np.float64),
  }


def _apply_ground_alignment(
  motion: _ArrayMotionLoader,
  cfg: Config,
  grounder: G1MotionGrounder | None,
) -> dict[str, np.ndarray]:
  """Optionally ground the resampled pose sequence and refresh root velocities."""
  if cfg.ground_alignment == "none":
    if grounder is None:
      raise RuntimeError("grounding QC requires an initialized G1 grounder.")
    report: GroundClearanceReport = grounder.measure(
      motion.motion_base_poss.detach().cpu().numpy(),
      motion.motion_base_rots.detach().cpu().numpy(),
      motion.motion_dof_poss.detach().cpu().numpy(),
    )
    zeros = np.zeros_like(report.min_distance)
    return _grounding_qc_arrays(
      cfg,
      GroundingResult(
        root_pos=motion.motion_base_poss.detach().cpu().numpy().copy(),
        correction=zeros,
        required_correction=np.maximum(
          0.0, cfg.ground_clearance_m - report.min_distance
        ),
        min_distance=report.min_distance,
        worst_geom=report.worst_geom,
      ),
    )
  if cfg.ground_alignment != "g1_collision":
    raise ValueError(f"Unsupported ground_alignment {cfg.ground_alignment!r}.")
  if grounder is None:
    raise RuntimeError("g1_collision alignment requires an initialized grounder.")

  result = grounder.ground(
    motion.motion_base_poss.detach().cpu().numpy(),
    motion.motion_base_rots.detach().cpu().numpy(),
    motion.motion_dof_poss.detach().cpu().numpy(),
    clearance=cfg.ground_clearance_m,
    smoothing=CorrectionSmoothing(
      output_fps=cfg.output_fps,
      smoothing_radius_s=cfg.ground_smoothing_radius_s,
    ),
  )
  corrected_root = torch.as_tensor(
    result.root_pos,
    dtype=motion.motion_base_poss.dtype,
    device=motion.motion_base_poss.device,
  )
  motion.motion_base_poss.copy_(corrected_root)
  # MotionLoader computed velocities before alignment. Recompute them now so both
  # the saved root/body velocities and the simulator state match the lifted path.
  motion._compute_velocities()
  return _grounding_qc_arrays(cfg, result)


def _replay(
  sim: Simulation,
  scene: Scene,
  robot_joint_indexes: list[int],
  motion: MotionLoader,
  frame_ids: torch.Tensor,
  root_states: torch.Tensor,
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
) -> dict[str, np.ndarray]:
  """Drive a complete reference sequence through batched forward kinematics.

  Positions and velocities are *written* to the model and only forward kinematics is
  evaluated, so this is a pure coordinate transform, not a physics rollout.
  """
  robot: Entity = scene["robot"]
  n = motion.output_frames
  batch_frames = int(frame_ids.shape[0])
  if batch_frames < 1:
    raise ValueError("Replay pool must contain at least one frame.")

  chunks: dict[str, list[torch.Tensor]] = {key: [] for key in _LOG_KEYS}
  for start in range(0, n, batch_frames):
    count = min(batch_frames, n - start)
    # MuJoCo-Warp environments are independent worlds. Treat a block of frames as
    # the environment batch; padding worlds repeat the last valid frame. Do not add
    # scene.env_origins: those are viewer-layout offsets, not motion data.
    source = torch.clamp(frame_ids + start, max=n - 1)
    root_states.copy_(robot.data.default_root_state)
    root_states[:, 0:3] = motion.motion_base_poss[source]
    root_states[:, 3:7] = motion.motion_base_rots[source]
    root_states[:, 7:10] = motion.motion_base_lin_vels[source]
    root_states[:, 10:13] = motion.motion_base_ang_vels[source]
    robot.write_root_state_to_sim(root_states)

    joint_pos.copy_(robot.data.default_joint_pos)
    joint_vel.copy_(robot.data.default_joint_vel)
    joint_pos[:, robot_joint_indexes] = motion.motion_dof_poss[source]
    joint_vel[:, robot_joint_indexes] = motion.motion_dof_vels[source]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    sim.forward()

    current = {
      "joint_pos": robot.data.joint_pos[:count],
      "joint_vel": robot.data.joint_vel[:count],
      "body_pos_w": robot.data.body_link_pos_w[:count],
      "body_quat_w": robot.data.body_link_quat_w[:count],
      "body_lin_vel_w": robot.data.body_link_lin_vel_w[:count],
      "body_ang_vel_w": robot.data.body_link_ang_vel_w[:count],
    }

    # Round-trip check for quaternion conventions and state layout. MuJoCo/Warp's
    # read-back differs by ~1.4e-5 m/s on fast LAFAN1 frames; 1e-3 still catches
    # every convention error by three orders of magnitude.
    stop = start + count
    torch.testing.assert_close(
      current["body_lin_vel_w"][:, 0],
      motion.motion_base_lin_vels[start:stop],
      atol=1e-3,
      rtol=1e-3,
    )
    torch.testing.assert_close(
      current["body_ang_vel_w"][:, 0],
      motion.motion_base_ang_vels[start:stop],
      atol=1e-3,
      rtol=1e-3,
    )
    for key, value in current.items():
      chunks[key].append(value.cpu())

  return {key: torch.cat(values).numpy() for key, values in chunks.items()}


def _safe_selected_path(input_dir: Path, value: object) -> Path:
  if not isinstance(value, str) or not value:
    raise TypeError(f"Selection paths must be non-empty strings, got {value!r}.")
  relative = Path(value)
  if relative.is_absolute() or ".." in relative.parts:
    raise ValueError(f"Selection path must stay below input_dir, got {value!r}.")
  path = input_dir / relative
  try:
    path.resolve().relative_to(input_dir.resolve())
  except ValueError as exc:
    raise ValueError(f"Selection path escapes input_dir: {value!r}.") from exc
  if path.suffix.lower() != ".csv":
    raise ValueError(f"Selection path is not a CSV: {value!r}.")
  if not path.is_file():
    raise FileNotFoundError(f"Selected motion does not exist: {path}")
  return path


def _motion_csv_files(input_dir: Path, selection: Path | None = None) -> list[Path]:
  """Return numeric motion CSVs, excluding sidecar metadata tables."""
  if selection is not None:
    with selection.open() as f:
      payload = json.load(f)
    values = payload.get("selected_files")
    if not isinstance(values, list):
      raise TypeError(f"{selection} must contain a selected_files list.")
    paths = [_safe_selected_path(input_dir, value) for value in values]
    if len(set(paths)) != len(paths):
      raise ValueError(f"{selection} contains duplicate selected_files entries.")
    return sorted(paths)
  return sorted(
    path for path in input_dir.rglob("*.csv") if path.name not in _NON_MOTION_CSV_NAMES
  )


def _read_source_motion(csv_path: Path, input_format: InputFormat) -> np.ndarray:
  skiprows = 0
  if input_format == "motiondecode":
    with csv_path.open(newline="") as f:
      header = tuple(next(csv.reader(f), ()))
    if header != MOTIONDECODE_HEADER:
      raise ValueError(
        f"{csv_path} has an incompatible MotionDecode header: expected "
        f"{len(MOTIONDECODE_HEADER)} columns, got {len(header)}."
      )
    skiprows = 1
  try:
    motion = np.loadtxt(
      csv_path, delimiter=",", dtype=np.float32, ndmin=2, skiprows=skiprows
    )
  except (OSError, UnicodeError, ValueError) as exc:
    raise ValueError(
      f"{csv_path} is not a valid numeric 36-column motion: {exc}"
    ) from exc
  if motion.ndim != 2 or motion.shape[1] != 36:
    raise ValueError(
      f"{csv_path} must have 36 numeric columns, got shape {motion.shape}."
    )
  if not np.isfinite(motion).all():
    raise ValueError(f"{csv_path} contains non-finite values.")
  return motion


def _relative_stem(csv_path: Path, input_dir: Path) -> Path:
  try:
    return csv_path.relative_to(input_dir).with_suffix("")
  except ValueError as exc:
    raise ValueError(f"{csv_path} is not below input_dir {input_dir}.") from exc


def _output_identity(
  csv_path: Path, input_dir: Path, output_dir: Path, source: str, preserve: bool
) -> tuple[str, Path]:
  relative_stem = _relative_stem(csv_path, input_dir)
  if preserve:
    name = f"{source}__{'__'.join(relative_stem.parts)}"
    return name, (output_dir / relative_stem).with_suffix(".npz")
  return f"{source}__{csv_path.stem}", output_dir / f"{source}__{csv_path.stem}.npz"


def _validate_unique_stems(csv_files: list[Path]) -> None:
  """Reject recursive inputs whose flat output names would overwrite one another."""
  by_stem: dict[str, Path] = {}
  conflicts: list[tuple[Path, Path]] = []
  for path in csv_files:
    previous = by_stem.setdefault(path.stem, path)
    if previous != path:
      conflicts.append((previous, path))
  if conflicts:
    examples = "; ".join(f"{a} <> {b}" for a, b in conflicts[:3])
    raise ValueError(
      f"Input contains {len(conflicts)} duplicate CSV stem(s), which would collide "
      f"in the flat output directory: {examples}. Use a pure motion subset."
    )


def _expected_output_frames(
  input_frames: int, input_fps: float, output_fps: float
) -> int:
  """Number of frames produced by mjlab's stop-exclusive resampling grid."""
  duration = (input_frames - 1) / input_fps
  return int(torch.arange(0, duration, 1.0 / output_fps, dtype=torch.float32).numel())


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> Any:
  if key not in data:
    raise ValueError(f"Existing clip is missing conversion metadata {key!r}.")
  return np.asarray(data[key]).reshape(-1)[0].item()


def _existing_num_frames(
  out_path: Path,
  cfg: Config,
  line_range: tuple[int, int],
  source_stat: os.stat_result,
) -> int:
  """Validate that a resumable clip was produced by this exact conversion."""
  start, stop = line_range
  expected = {
    "converter_schema_version": _CONVERTER_SCHEMA_VERSION,
    "input_format": cfg.input_format,
    "input_fps": cfg.input_fps,
    "fps": cfg.output_fps,
    "ground_alignment": cfg.ground_alignment,
    "ground_clearance_m": cfg.ground_clearance_m,
    "ground_smoothing_radius_s": cfg.ground_smoothing_radius_s,
    "line_start": start,
    "line_stop": stop,
    "source_size": source_stat.st_size,
    "source_mtime_ns": source_stat.st_mtime_ns,
  }
  with np.load(out_path) as data:
    for key, wanted in expected.items():
      actual = _scalar(data, key)
      matches = (
        math.isclose(float(actual), float(wanted))
        if key
        in {
          "input_fps",
          "fps",
          "ground_clearance_m",
          "ground_smoothing_radius_s",
        }
        else actual == wanted
      )
      if not matches:
        raise ValueError(
          f"{out_path} has {key}={actual!r}, expected {wanted!r}; use a clean "
          "output directory or --overwrite True."
        )

    # Schema v4 clips always carry a complete, finite QC record. No-op conversions
    # still measure the active collision geometry but record zero correction.
    for key in _GROUND_QC_KEYS:
      _scalar(data, key)

    missing = [key for key in _LOG_KEYS if key not in data]
    if missing:
      raise ValueError(f"{out_path} is missing motion arrays {missing}.")
    num_frames = int(data["joint_pos"].shape[0])
    wanted_frames = _expected_output_frames(
      stop - start + 1, cfg.input_fps, cfg.output_fps
    )
    if num_frames != wanted_frames:
      raise ValueError(
        f"{out_path} has {num_frames} frames, expected {wanted_frames} for input "
        f"rows {start}:{stop}."
      )
    for key in _LOG_KEYS:
      if data[key].shape[0] != num_frames:
        raise ValueError(
          f"{out_path} array {key!r} has {data[key].shape[0]} frames, "
          f"expected {num_frames}."
        )
  return num_frames


def _write_npz(out_path: Path, **arrays: np.ndarray) -> None:
  """Write a clip beside its destination, then atomically replace the destination."""
  temporary = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("wb") as f:
      np.savez(f, **arrays)
    os.replace(temporary, out_path)
  finally:
    temporary.unlink(missing_ok=True)


def main(cfg: Config) -> None:
  if cfg.input_fps <= 0.0 or cfg.output_fps <= 0.0:
    raise ValueError("input_fps and output_fps must both be positive.")
  if cfg.ground_alignment not in {"none", "g1_collision"}:
    raise ValueError(
      "ground_alignment must be either 'none' or 'g1_collision', got "
      f"{cfg.ground_alignment!r}."
    )
  if not math.isfinite(cfg.ground_clearance_m) or cfg.ground_clearance_m < 0.0:
    raise ValueError("ground_clearance_m must be finite and non-negative.")
  if (
    not math.isfinite(cfg.ground_smoothing_radius_s)
    or cfg.ground_smoothing_radius_s < 0.0
  ):
    raise ValueError("ground_smoothing_radius_s must be finite and non-negative.")
  if cfg.replay_batch_frames < 1:
    raise ValueError("replay_batch_frames must be at least 1.")
  if cfg.limit is not None and cfg.limit < 1:
    raise ValueError("limit must be at least 1 when provided.")
  if cfg.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
      "CUDA requested but unavailable. Pass --device cpu explicitly if that is "
      "really what you want (it is roughly 50x slower)."
    )
  if cfg.input_format == "bones-seed" and not math.isclose(cfg.input_fps, 120.0):
    raise ValueError("BONES-SEED must be converted with --input-fps 120.")
  if cfg.input_format == "motiondecode" and not math.isclose(cfg.input_fps, 120.0):
    raise ValueError("MotionDecode must be converted with --input-fps 120.")
  if cfg.manifest_checkpoint_files < 1:
    raise ValueError("manifest_checkpoint_files must be at least 1.")

  input_dir = Path(cfg.input_dir)
  selection_path = Path(cfg.selection) if cfg.selection is not None else None
  csv_files = _motion_csv_files(input_dir, selection_path)
  if not csv_files:
    raise FileNotFoundError(f"No .csv files under {input_dir.resolve()}")
  if not cfg.preserve_relative_paths:
    _validate_unique_stems(csv_files)
  if cfg.limit is not None:
    csv_files = csv_files[: cfg.limit]

  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest_path = Path(cfg.manifest)
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  partial_manifest_path = manifest_path.with_suffix(f"{manifest_path.suffix}.partial")
  invalid_manifest_path = (
    Path(cfg.invalid_manifest)
    if cfg.invalid_manifest is not None
    else manifest_path.with_name(f"{manifest_path.stem}.invalid.json")
  )

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / cfg.output_fps
  scene_cfg = unitree_g1_flat_tracking_env_cfg().scene
  # Reuse the exact robot asset used by Ex-GRMT training, including its InstinctMJ
  # foot/knee collision primitives. The upstream tracking scene uses a different G1
  # collision profile and would silently make replay/body FK disagree with grounding.
  scene_cfg.entities["robot"] = get_ex_grmt_g1_robot_cfg()
  scene_cfg.num_envs = cfg.replay_batch_frames
  scene = Scene(scene_cfg, device=cfg.device)
  model = scene.compile()
  sim = Simulation(
    num_envs=cfg.replay_batch_frames, cfg=sim_cfg, model=model, device=cfg.device
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(list(G1_JOINT_ORDER), preserve_order=True)[0]
  frame_ids = torch.arange(cfg.replay_batch_frames, device=cfg.device)
  root_states = robot.data.default_root_state.clone()
  joint_pos = robot.data.default_joint_pos.clone()
  joint_vel = robot.data.default_joint_vel.clone()
  # Grounding metadata is a finite, independently auditable measurement even when
  # alignment is disabled, so both modes use the exact training collision profile.
  grounder = G1MotionGrounder()

  if cfg.append and manifest_path.exists():
    with manifest_path.open() as f:
      clips = json.load(f)["clips"]
    clips_by_name = {clip["name"]: clip for clip in clips}
    if len(clips_by_name) != len(clips):
      raise ValueError(f"{manifest_path} contains duplicate clip names.")
  else:
    clips_by_name: dict[str, dict[str, Any]] = {}
  invalid_sources: list[dict[str, str]] = []
  provenance = _conversion_provenance(cfg, input_dir, selection_path)

  try:
    for file_index, csv_path in enumerate(
      tqdm(csv_files, desc="sequences", unit="file"), start=1
    ):
      stat_before = csv_path.stat()
      try:
        source_motion = _read_source_motion(csv_path, cfg.input_format)
      except ValueError as exc:
        if not cfg.skip_invalid_sources:
          raise
        relative = _relative_stem(csv_path, input_dir).with_suffix(".csv")
        invalid_sources.append({"path": relative.as_posix(), "error": str(exc)})
        tqdm.write(f"[ex-grmt] skipped invalid source {relative}: {exc}")
        continue
      source_stat = csv_path.stat()
      if (
        stat_before.st_size != source_stat.st_size
        or stat_before.st_mtime_ns != source_stat.st_mtime_ns
      ):
        raise RuntimeError(f"Source changed while it was being read: {csv_path}")
      line_range = (1, int(source_motion.shape[0]))
      name, out_path = _output_identity(
        csv_path,
        input_dir,
        output_dir,
        cfg.source,
        cfg.preserve_relative_paths,
      )
      out_path.parent.mkdir(parents=True, exist_ok=True)

      if out_path.exists() and not cfg.overwrite:
        num_frames = _existing_num_frames(out_path, cfg, line_range, source_stat)
        clips_by_name[name] = _entry(name, cfg, out_path, manifest_path, num_frames)
      else:
        motion = _ArrayMotionLoader(
          source_motion,
          input_fps=cfg.input_fps,
          output_fps=cfg.output_fps,
          device=cfg.device,
          input_format=cfg.input_format,
        )
        grounding_qc = _apply_ground_alignment(motion, cfg, grounder)
        log = _replay(
          sim,
          scene,
          robot_joint_indexes,
          motion,
          frame_ids,
          root_states,
          joint_pos,
          joint_vel,
        )
        _write_npz(
          out_path,
          converter_schema_version=np.array(
            [_CONVERTER_SCHEMA_VERSION], dtype=np.int64
          ),
          fps=np.array([cfg.output_fps]),
          input_fps=np.array([cfg.input_fps]),
          input_format=np.array([cfg.input_format]),
          line_start=np.array([line_range[0]], dtype=np.int64),
          line_stop=np.array([line_range[1]], dtype=np.int64),
          source_size=np.array([source_stat.st_size], dtype=np.int64),
          source_mtime_ns=np.array([source_stat.st_mtime_ns], dtype=np.int64),
          **grounding_qc,
          **log,
        )
        clips_by_name[name] = _entry(
          name, cfg, out_path, manifest_path, log["joint_pos"].shape[0]
        )

      if file_index % cfg.manifest_checkpoint_files == 0:
        _write_manifest(partial_manifest_path, list(clips_by_name.values()), provenance)
        _write_invalid_manifest(invalid_manifest_path, invalid_sources)
  except BaseException:
    # Preserve a previously complete manifest. The partial manifest is diagnostic;
    # resume reconstructs entries from fingerprinted NPZ files in the normal loop.
    _write_manifest(partial_manifest_path, list(clips_by_name.values()), provenance)
    _write_invalid_manifest(invalid_manifest_path, invalid_sources)
    raise
  else:
    _write_manifest(manifest_path, list(clips_by_name.values()), provenance)
    _write_invalid_manifest(invalid_manifest_path, invalid_sources)
    partial_manifest_path.unlink(missing_ok=True)

  clips = list(clips_by_name.values())
  total_frames = sum(c["num_frames"] for c in clips)

  print(
    f"[ex-grmt] wrote {len(clips)} complete sequences "
    f"({total_frames / cfg.output_fps / 60:.1f} min) to {manifest_path}"
  )


def _conversion_provenance(
  cfg: Config, input_dir: Path, selection_path: Path | None
) -> dict[str, Any]:
  provenance: dict[str, Any] = {
    "dataset": cfg.source,
    "input_dir": str(input_dir.resolve()),
    "input_format": cfg.input_format,
    "input_fps": float(cfg.input_fps),
    "output_fps": float(cfg.output_fps),
    "converter_schema_version": _CONVERTER_SCHEMA_VERSION,
    "ground_alignment": cfg.ground_alignment,
    "ground_clearance_m": float(cfg.ground_clearance_m),
    "ground_smoothing_radius_s": float(cfg.ground_smoothing_radius_s),
  }
  if selection_path is not None:
    provenance["selection"] = str(selection_path.resolve())
    provenance["selection_sha256"] = hashlib.sha256(
      selection_path.read_bytes()
    ).hexdigest()
  if cfg.input_format == "motiondecode":
    provenance["paper_status"] = (
      "MotionDecode extension/proxy; not an Extreme-RGMT Table IV source."
    )
  return provenance


def _write_manifest(
  manifest_path: Path,
  clips: list[dict[str, Any]],
  provenance: dict[str, Any],
) -> None:
  """Atomically replace a manifest so interruption cannot leave truncated JSON."""
  clips.sort(key=lambda clip: clip["name"])
  total_frames = sum(int(clip["num_frames"]) for clip in clips)
  total_seconds = sum(int(clip["num_frames"]) / float(clip["fps"]) for clip in clips)
  temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
  try:
    with temporary.open("w") as f:
      json.dump(
        {
          "kind": "complete_sequences",
          "provenance": provenance,
          "clip_count": len(clips),
          "total_frames": total_frames,
          "total_seconds": total_seconds,
          "clips": clips,
        },
        f,
        indent=2,
      )
    os.replace(temporary, manifest_path)
  finally:
    temporary.unlink(missing_ok=True)


def _write_invalid_manifest(
  manifest_path: Path, invalid_sources: list[dict[str, str]]
) -> None:
  manifest_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
  try:
    with temporary.open("w") as f:
      json.dump(
        {"kind": "invalid_motion_sources", "sources": invalid_sources}, f, indent=2
      )
    os.replace(temporary, manifest_path)
  finally:
    temporary.unlink(missing_ok=True)


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
