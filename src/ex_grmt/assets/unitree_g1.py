"""Unitree G1 with the InstinctMJ Perceptive Shadowing collision primitives.

The collision dimensions below are adapted from Project Instinct's
``g1_29dof_torsobase_popsicle.xml`` at commit
``4ed2b32f8719ff9fc138708341031e935afda0d2``:

https://github.com/project-instinct/InstinctMJ/blob/4ed2b32f8719ff9fc138708341031e935afda0d2/src/instinct_mj/assets/resources/unitree_g1/xml/g1_29dof_torsobase_popsicle.xml

InstinctMJ is licensed under CC BY-NC 4.0. This adaptation renames the otherwise
unnamed collision geoms so mjlab contact sensors can address them, and attaches the
same primitives to mjlab's pelvis-root G1. The latter deliberately preserves Ex-GRMT's
joint order, motion files, and policy I/O; InstinctMJ's complete torso-root MJCF is not
checkpoint- or motion-compatible with those conventions.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import mujoco
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg

HIP_PITCH_EXPR = ".*_hip_pitch_joint"

MJLAB_HIP_PITCH_EFFORT_LIMIT = 88.0
"""mjlab's hip-pitch peak torque, N*m. Matches the G1 hardware sheet."""

SONIC_HIP_PITCH_EFFORT_LIMIT = 139.0
"""Hip-pitch peak torque used by SONIC's *training* config, N*m.

mjlab groups hip pitch with hip yaw and waist yaw on the 88 N*m actuator, which is
what the G1 hardware sheet says (SONIC's own deployment config lists the same
88 N*m / 32 rad/s). SONIC's training config -- adapted from BeyondMimic, the lineage
this paper builds on -- instead drives hip pitch at 139 N*m, like hip roll and knee.

Scope of the override: it raises the **torque clamp only**. Expanding mjlab's action
term,

    tau = kp * (q_ref + scale * a - q) - kd * qdot
        = kp * e  +  kp * scale * a  -  kd * qdot,   scale = 0.25 * effort / kp
        = kp * e  +  0.25 * effort * a  -  kd * qdot

the action's authority is ``0.25 * effort``. Re-deriving ``scale`` from a raised limit
would therefore change what one unit of policy output means and silently reinterpret
every trained checkpoint, so ``env_cfg`` keeps mjlab's frozen ``G1_ACTION_SCALE`` and
the per-unit authority stays at 22 N*m. Measured peak hip-pitch torque over a backflip
takeoff is 22 N*m against the 88 N*m limit, so today's policy never reaches the clamp
either -- this is headroom for a run that pushes harder, not a change to the current
one. ``kp`` is left alone deliberately: SONIC's matching ``kd`` would cost 72.6 N*m of
damping at the 11.5 rad/s the takeoff reaches, against 29.4 N*m here.
"""


def hip_pitch_effort_limit() -> float:
  """Resolve the hip-pitch effort limit, mjlab's value unless overridden.

  Override with ``EX_GRMT_HIP_PITCH_EFFORT`` (N*m); see
  :data:`SONIC_HIP_PITCH_EFFORT_LIMIT` for what it does and does not change.
  """
  raw = os.environ.get("EX_GRMT_HIP_PITCH_EFFORT")
  if raw is None:
    return MJLAB_HIP_PITCH_EFFORT_LIMIT
  value = float(raw)
  if value <= 0.0:
    raise ValueError(f"EX_GRMT_HIP_PITCH_EFFORT must be positive, got {value}.")
  return value


_SILVER = (0.7, 0.7, 0.7, 1.0)
_BLACK = (0.2, 0.2, 0.2, 1.0)
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
_SIDEWAYS_QUAT = (0.707105, 0.0, -0.707108, 0.0)


@dataclass(frozen=True)
class _CollisionGeom:
  body: str
  name: str
  geom_type: mujoco.mjtGeom
  size: tuple[float, ...]
  pos: tuple[float, float, float]
  quat: tuple[float, float, float, float] = _IDENTITY_QUAT
  rgba: tuple[float, float, float, float] = _SILVER


def _foot_geoms(side: str) -> tuple[_CollisionGeom, ...]:
  body = f"{side}_ankle_roll_link"
  return (
    _CollisionGeom(
      body,
      f"{side}_foot1_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.01, 0.025),
      (0.075, -0.026, -0.025),
      (0.707105, 0.0, 0.707108, 0.0),
      _BLACK,
    ),
    _CollisionGeom(
      body,
      f"{side}_foot2_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.008, 0.0835),
      (0.0395, -0.018, -0.025),
      _SIDEWAYS_QUAT,
      _BLACK,
    ),
    _CollisionGeom(
      body,
      f"{side}_foot3_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.01, 0.091),
      (0.039, -0.01, -0.025),
      _SIDEWAYS_QUAT,
      _BLACK,
    ),
    _CollisionGeom(
      body,
      f"{side}_foot4_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.01, 0.093),
      (0.039, 0.0, -0.025),
      _SIDEWAYS_QUAT,
      _BLACK,
    ),
    _CollisionGeom(
      body,
      f"{side}_foot5_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.01, 0.091),
      (0.039, 0.01, -0.025),
      _SIDEWAYS_QUAT,
      _BLACK,
    ),
    _CollisionGeom(
      body,
      f"{side}_foot6_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.008, 0.0835),
      (0.0395, 0.018, -0.025),
      _SIDEWAYS_QUAT,
      _BLACK,
    ),
    _CollisionGeom(
      body,
      f"{side}_foot7_collision",
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      (0.01, 0.025),
      (0.075, 0.026, -0.025),
      (0.707105, 0.0, 0.707108, 0.0),
      _BLACK,
    ),
  )


_INSTINCT_COLLISION_GEOMS = (
  # The four transverse capsules are the "popsicle" torso used by Perceptive
  # Shadowing. They cover the torso/head much more closely than one large capsule.
  _CollisionGeom(
    "torso_link",
    "torso1_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.073, 0.032),
    (0.005, 0.0, 0.22),
    (0.707105, 0.707108, 0.0, 0.0),
  ),
  _CollisionGeom(
    "torso_link",
    "torso2_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.07, 0.028),
    (0.005, 0.0, 0.13),
    (0.707105, 0.707108, 0.0, 0.0),
  ),
  _CollisionGeom(
    "torso_link",
    "torso3_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.065, 0.02),
    (0.005, 0.0, 0.06),
    (0.707105, 0.707108, 0.0, 0.0),
  ),
  _CollisionGeom(
    "torso_link",
    "torso4_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.068, 0.005),
    (0.01, 0.0, 0.415),
    (-3.67321e-06, 1.0, 0.0, 0.0),
  ),
  _CollisionGeom(
    "pelvis",
    "pelvis_collision",
    mujoco.mjtGeom.mjGEOM_SPHERE,
    (0.07,),
    (0.0, 0.0, -0.08),
    rgba=_BLACK,
  ),
  _CollisionGeom(
    "left_hip_roll_link",
    "left_hip_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.05, 0.1),
    (0.02, 0.0, -0.1),
  ),
  _CollisionGeom(
    "left_knee_link",
    "left_knee_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.05, 0.1),
    (0.02, 0.0, -0.1),
  ),
  *_foot_geoms("left"),
  _CollisionGeom(
    "right_hip_roll_link",
    "right_hip_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.05, 0.1),
    (0.02, 0.0, -0.1),
  ),
  _CollisionGeom(
    "right_knee_link",
    "right_knee_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.05, 0.1),
    (0.02, 0.0, -0.1),
  ),
  *_foot_geoms("right"),
  _CollisionGeom(
    "left_shoulder_yaw_link",
    "left_shoulder_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.035, 0.065),
    (0.0, 0.0, -0.015),
    (-3.67321e-06, 1.0, 0.0, 0.0),
  ),
  _CollisionGeom(
    "left_elbow_link",
    "left_elbow_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.035, 0.065),
    (0.055, 0.0, -0.01),
    _SIDEWAYS_QUAT,
  ),
  _CollisionGeom(
    "left_wrist_yaw_link",
    "left_hand_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.05, 0.025),
    (0.075, 0.0, 0.0),
    _SIDEWAYS_QUAT,
  ),
  _CollisionGeom(
    "right_shoulder_yaw_link",
    "right_shoulder_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.035, 0.065),
    (0.0, 0.0, -0.015),
    (-3.67321e-06, 1.0, 0.0, 0.0),
  ),
  _CollisionGeom(
    "right_elbow_link",
    "right_elbow_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.035, 0.065),
    (0.055, 0.0, -0.01),
    _SIDEWAYS_QUAT,
  ),
  _CollisionGeom(
    "right_wrist_yaw_link",
    "right_hand_collision",
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    (0.05, 0.025),
    (0.075, 0.0, 0.0),
    _SIDEWAYS_QUAT,
  ),
)


def get_instinct_collision_g1_spec() -> mujoco.MjSpec:
  """Return mjlab's pelvis-root G1 with InstinctMJ's collision primitives."""
  upstream_cfg = get_g1_robot_cfg()
  spec = upstream_cfg.spec_fn()

  # mjlab's source XML currently has 33 named, active collision geoms. Treat an
  # upstream asset-layout change as an error instead of silently producing a hybrid.
  active_geoms = [
    geom for geom in spec.geoms if geom.contype != 0 or geom.conaffinity != 0
  ]
  if len(active_geoms) != 33 or any(
    not geom.name.endswith("_collision") for geom in active_geoms
  ):
    names = tuple(geom.name for geom in active_geoms)
    raise RuntimeError(
      "mjlab's G1 collision layout changed; refusing to mix it with the "
      f"InstinctMJ profile. Active geoms: {names}"
    )
  for geom in active_geoms:
    spec.delete(geom)

  for geom in _INSTINCT_COLLISION_GEOMS:
    body = spec.body(geom.body)
    if body is None:
      raise RuntimeError(f"mjlab's G1 no longer contains body {geom.body!r}.")
    body.add_geom(
      name=geom.name,
      type=geom.geom_type,
      size=geom.size,
      pos=geom.pos,
      quat=geom.quat,
      contype=1,
      conaffinity=1,
      group=3,
      rgba=geom.rgba,
    )
  return spec


def _split_hip_pitch(articulation, effort_limit: float):
  """Give hip pitch its own actuator entry so its effort limit can differ.

  mjlab packs hip pitch, hip yaw and waist yaw into one entry, and
  ``EntityCfg.sort_actuators`` is False, so a second entry moves hip pitch to the end
  of that block in ``ctrl`` order. Per-joint gains, limits and armature are unchanged
  (``tests/test_g1_asset.py``) and mjlab addresses actuators by name, but the default
  path refuses to take that risk for nothing: at mjlab's own limit the split would buy
  no behaviour change at all, so the articulation is returned untouched.
  """
  if effort_limit == MJLAB_HIP_PITCH_EFFORT_LIMIT:
    return articulation

  actuators = []
  found = False
  for actuator in articulation.actuators:
    names = tuple(actuator.target_names_expr)
    if HIP_PITCH_EXPR not in names:
      actuators.append(actuator)
      continue
    found = True
    rest = tuple(n for n in names if n != HIP_PITCH_EXPR)
    if rest:
      actuators.append(dataclasses.replace(actuator, target_names_expr=rest))
    actuators.append(
      dataclasses.replace(
        actuator,
        target_names_expr=(HIP_PITCH_EXPR,),
        effort_limit=effort_limit,
      )
    )
  if not found:
    raise RuntimeError(
      f"mjlab's G1 no longer declares an actuator for {HIP_PITCH_EXPR!r}; refusing to "
      "silently leave the hip-pitch effort limit unset."
    )
  return dataclasses.replace(articulation, actuators=tuple(actuators))


def get_ex_grmt_g1_robot_cfg() -> EntityCfg:
  """Return the Ex-GRMT G1 while preserving mjlab's dynamics and joint order."""
  cfg = get_g1_robot_cfg()
  cfg.spec_fn = get_instinct_collision_g1_spec
  cfg.articulation = _split_hip_pitch(cfg.articulation, hip_pitch_effort_limit())
  return cfg
