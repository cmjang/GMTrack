"""Ground G1 reference poses against the robot's collision geometry.

The visual meshes are deliberately not used here.  Their decorative surfaces do
not define simulation contact, while the InstinctMJ collision profile is the shape
the policy actually has to keep above the floor.  A :class:`G1MotionGrounder`
compiles that profile together with a z=0 plane once and can then inspect or correct
many motion clips.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ex_grmt.assets import get_instinct_collision_g1_spec

G1_JOINT_ORDER = (
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

G1_FK_BODY_ORDER = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

_GROUND_PLANE_NAME = "motion_grounding_plane"
DEFAULT_CLEARANCE = 0.003


@dataclass(frozen=True)
class CorrectionSmoothing:
  """Time-domain parameters for a conservative correction envelope.

  ``smoothing_radius_s`` controls how far an isolated penetration is spread to each
  side before Gaussian smoothing.  If ``gaussian_sigma_s`` is omitted, one third of
  the radius is used.  All values are seconds so changing the output frame rate does
  not change the physical-time filter.
  """

  output_fps: float = 50.0
  smoothing_radius_s: float = 0.3
  gaussian_sigma_s: float | None = None


DEFAULT_SMOOTHING = CorrectionSmoothing()


@dataclass(frozen=True)
class GroundClearanceReport:
  """Per-frame distance from the lowest active robot geom to the z=0 plane."""

  min_distance: NDArray[np.float64]
  """Signed distance in metres; a negative value means penetration."""

  worst_geom: tuple[str, ...]
  """Name of the geom attaining ``min_distance`` in each frame."""


@dataclass(frozen=True)
class GroundingResult:
  """Upward-only correction and the resulting root translations."""

  root_pos: NDArray[np.float64]
  """Copy of the input root positions with ``correction`` added to z."""

  correction: NDArray[np.float64]
  """Applied translation; possibly smoothed, and always >= required correction."""

  required_correction: NDArray[np.float64]
  """Exact ``max(0, clearance - min_distance)`` before optional smoothing."""

  min_distance: NDArray[np.float64]
  """Signed distance before correction."""

  worst_geom: tuple[str, ...]
  """Lowest active robot collision geom before correction."""


def _validate_fk_layout(model: mujoco.MjModel) -> None:
  body_names = tuple(model.body(i).name for i in range(1, model.nbody))
  if body_names != G1_FK_BODY_ORDER:
    raise RuntimeError(
      "G1 MuJoCo FK body ordering changed; grounding cannot safely interpret the "
      f"motion convention. Expected {G1_FK_BODY_ORDER}, got {body_names}."
    )

  joint_names = tuple(model.joint(i).name for i in range(1, model.njnt))
  if joint_names != G1_JOINT_ORDER:
    raise RuntimeError(
      "G1 MuJoCo joint ordering changed; grounding cannot safely write the "
      f"29-DoF motion block. Expected {G1_JOINT_ORDER}, got {joint_names}."
    )

  if (
    model.njnt != len(G1_JOINT_ORDER) + 1
    or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
    or model.jnt_qposadr[0] != 0
  ):
    raise RuntimeError("G1 grounding model no longer has the expected root free joint.")
  joint_types = model.jnt_type[1:]
  if np.any(joint_types != mujoco.mjtJoint.mjJNT_HINGE):
    raise RuntimeError("G1 grounding model contains a non-hinge actuated joint.")


def _frames(value: ArrayLike, name: str, width: int) -> NDArray[np.float64]:
  try:
    array = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError) as exc:
    raise TypeError(f"{name} must be a numeric array with shape (F, {width}).") from exc
  if array.ndim != 2 or array.shape[1] != width:
    raise ValueError(f"{name} must have shape (F, {width}), got {array.shape}.")
  if array.shape[0] == 0:
    raise ValueError(f"{name} must contain at least one frame.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} contains a non-finite value.")
  return array


def _finite_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
  try:
    array = np.asarray(value, dtype=np.float64)
  except (TypeError, ValueError) as exc:
    raise TypeError(f"{name} must be a numeric one-dimensional array.") from exc
  if array.ndim != 1:
    raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}.")
  if not np.all(np.isfinite(array)):
    raise ValueError(f"{name} contains a non-finite value.")
  return array


def required_correction(
  min_distance: ArrayLike, *, clearance: float = DEFAULT_CLEARANCE
) -> NDArray[np.float64]:
  """Return the exact upward translation needed to reach ``clearance``."""
  distances = _finite_vector(min_distance, "min_distance")
  if not np.isfinite(clearance) or clearance < 0.0:
    raise ValueError(f"clearance must be finite and non-negative, got {clearance}.")
  return np.maximum(0.0, clearance - distances)


def smooth_correction_upper_envelope(
  correction: ArrayLike,
  *,
  output_fps: float = 50.0,
  smoothing_radius_s: float = 0.3,
  gaussian_sigma_s: float | None = None,
) -> NDArray[np.float64]:
  """Smooth a required correction without ever reducing it.

  A maximum filter first spreads penetration corrections through a time radius, then
  a Gaussian removes the blocky shoulders.  The final pointwise maximum with the
  input is load-bearing: Gaussian smoothing alone can undershoot sharp penetration
  peaks and put collision geometry back below the requested clearance.
  """
  required = _finite_vector(correction, "correction")
  if np.any(required < 0.0):
    raise ValueError("correction must be non-negative.")
  if not np.isfinite(output_fps) or output_fps <= 0.0:
    raise ValueError(f"output_fps must be finite and positive, got {output_fps}.")
  if not np.isfinite(smoothing_radius_s) or smoothing_radius_s < 0.0:
    raise ValueError(
      f"smoothing_radius_s must be finite and non-negative, got {smoothing_radius_s}."
    )
  sigma_s = smoothing_radius_s / 3.0 if gaussian_sigma_s is None else gaussian_sigma_s
  if not np.isfinite(sigma_s) or sigma_s < 0.0:
    raise ValueError(
      f"gaussian_sigma_s must be finite and non-negative, got {sigma_s}."
    )
  if required.size == 0 or smoothing_radius_s == 0.0:
    return required.copy()

  radius_frames = int(np.ceil(smoothing_radius_s * output_fps))
  padded = np.pad(required, radius_frames, mode="edge")
  windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius_frames + 1)
  maximum_envelope = np.max(windows, axis=1)

  sigma_frames = sigma_s * output_fps
  if sigma_frames == 0.0:
    smoothed = maximum_envelope
  else:
    gaussian_radius = max(1, int(np.ceil(4.0 * sigma_frames)))
    offsets = np.arange(-gaussian_radius, gaussian_radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_frames))
    kernel /= np.sum(kernel)
    smoothed = np.convolve(
      np.pad(maximum_envelope, gaussian_radius, mode="edge"), kernel, mode="valid"
    )
  return np.maximum(required, smoothed)


class G1MotionGrounder:
  """Measure and remove G1 collision-geometry penetration frame by frame.

  The instance owns a mutable :class:`mujoco.MjData` scratch buffer and is therefore
  reusable but not thread-safe.
  """

  def __init__(self) -> None:
    spec = get_instinct_collision_g1_spec()
    spec.worldbody.add_geom(
      name=_GROUND_PLANE_NAME,
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(0.0, 0.0, 0.1),
      pos=(0.0, 0.0, 0.0),
      contype=1,
      conaffinity=1,
    )
    self.model = spec.compile()
    _validate_fk_layout(self.model)
    self.data = mujoco.MjData(self.model)

    self._plane_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_GEOM, _GROUND_PLANE_NAME
    )
    self._robot_geom_ids = tuple(
      geom_id
      for geom_id in range(self.model.ngeom)
      if self.model.geom_bodyid[geom_id] != 0
      and (
        self.model.geom_contype[geom_id] != 0
        or self.model.geom_conaffinity[geom_id] != 0
      )
    )
    if len(self._robot_geom_ids) != 29:
      names = tuple(self.model.geom(i).name for i in self._robot_geom_ids)
      raise RuntimeError(
        "Instinct G1 active collision layout changed; expected 29 robot geoms, "
        f"got {len(names)}: {names}."
      )
    self._robot_geom_names = tuple(
      self.model.geom(i).name for i in self._robot_geom_ids
    )
    if any(not name.endswith("_collision") for name in self._robot_geom_names):
      raise RuntimeError(
        "Instinct G1 active robot geoms contain an unexpected non-collision geom: "
        f"{self._robot_geom_names}."
      )
    self._joint_qpos_adr = self.model.jnt_qposadr[1:].copy()

  def _validate_motion(
    self,
    root_pos: ArrayLike,
    root_quat_wxyz: ArrayLike,
    joint_pos: ArrayLike,
  ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    roots = _frames(root_pos, "root_pos", 3)
    quats = _frames(root_quat_wxyz, "root_quat_wxyz", 4)
    joints = _frames(joint_pos, "joint_pos", len(G1_JOINT_ORDER))
    frame_counts = (roots.shape[0], quats.shape[0], joints.shape[0])
    if len(set(frame_counts)) != 1:
      raise ValueError(
        "root_pos, root_quat_wxyz, and joint_pos must have the same frame count, "
        f"got {frame_counts}."
      )
    quat_norm = np.linalg.norm(quats, axis=1)
    if np.any(quat_norm == 0.0):
      raise ValueError("root_quat_wxyz contains a zero-length quaternion.")
    return roots, quats / quat_norm[:, None], joints

  def measure(
    self,
    root_pos: ArrayLike,
    root_quat_wxyz: ArrayLike,
    joint_pos: ArrayLike,
  ) -> GroundClearanceReport:
    """Return the lowest signed plane distance and geom for every frame."""
    roots, quats, joints = self._validate_motion(root_pos, root_quat_wxyz, joint_pos)
    min_distance = np.empty(roots.shape[0], dtype=np.float64)
    worst_geom: list[str] = []

    for frame in range(roots.shape[0]):
      self.data.qpos[:] = self.model.qpos0
      self.data.qpos[0:3] = roots[frame]
      self.data.qpos[3:7] = quats[frame]
      self.data.qpos[self._joint_qpos_adr] = joints[frame]
      mujoco.mj_forward(self.model, self.data)

      distances = np.fromiter(
        (
          mujoco.mj_geomDistance(
            self.model,
            self.data,
            self._plane_id,
            geom_id,
            np.inf,
            None,
          )
          for geom_id in self._robot_geom_ids
        ),
        dtype=np.float64,
        count=len(self._robot_geom_ids),
      )
      worst_index = int(np.argmin(distances))
      min_distance[frame] = distances[worst_index]
      worst_geom.append(self._robot_geom_names[worst_index])

    return GroundClearanceReport(min_distance, tuple(worst_geom))

  def ground(
    self,
    root_pos: ArrayLike,
    root_quat_wxyz: ArrayLike,
    joint_pos: ArrayLike,
    *,
    clearance: float = DEFAULT_CLEARANCE,
    smoothing: CorrectionSmoothing | None = DEFAULT_SMOOTHING,
  ) -> GroundingResult:
    """Raise penetrating frames so all active collision geoms clear the plane."""
    roots = _frames(root_pos, "root_pos", 3)
    report = self.measure(roots, root_quat_wxyz, joint_pos)
    required = required_correction(report.min_distance, clearance=clearance)
    correction = required
    if smoothing is not None:
      correction = smooth_correction_upper_envelope(
        required,
        output_fps=smoothing.output_fps,
        smoothing_radius_s=smoothing.smoothing_radius_s,
        gaussian_sigma_s=smoothing.gaussian_sigma_s,
      )
    corrected = roots.copy()
    corrected[:, 2] += correction
    return GroundingResult(
      root_pos=corrected,
      correction=correction,
      required_correction=required,
      min_distance=report.min_distance,
      worst_geom=report.worst_geom,
    )
