"""Checks for the InstinctMJ Perceptive Shadowing collision profile."""

from __future__ import annotations

import mujoco
import pytest

from gmtrack.assets import get_instinct_collision_g1_spec
from gmtrack.scripts.prepare_motions import G1_JOINT_ORDER


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return get_instinct_collision_g1_spec().compile()


def _geom(model: mujoco.MjModel, name: str) -> int:
  geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
  assert geom_id >= 0, f"missing collision geom {name}"
  return geom_id


def test_instinct_collision_profile_preserves_gmtrack_kinematics(model):
  """Only collisions change; motion tensors and policy I/O keep their joint order."""
  assert model.body(1).name == "pelvis"
  assert tuple(model.joint(i).name for i in range(1, model.njnt)) == G1_JOINT_ORDER


def test_instinct_collision_profile_has_named_sensor_addressable_geoms(model):
  active_names = tuple(
    model.geom(i).name
    for i in range(model.ngeom)
    if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
  )
  assert len(active_names) == 29
  assert all(name.endswith("_collision") for name in active_names)
  assert sum(name.startswith("left_foot") for name in active_names) == 7
  assert sum(name.startswith("right_foot") for name in active_names) == 7


def test_instinct_popsicle_torso_geometry(model):
  expected = {
    "torso1_collision": ((0.073, 0.032), (0.005, 0.0, 0.22)),
    "torso2_collision": ((0.07, 0.028), (0.005, 0.0, 0.13)),
    "torso3_collision": ((0.065, 0.02), (0.005, 0.0, 0.06)),
    "torso4_collision": ((0.068, 0.005), (0.01, 0.0, 0.415)),
  }
  for name, (size, pos) in expected.items():
    geom_id = _geom(model, name)
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert model.geom_size[geom_id, :2] == pytest.approx(size)
    assert model.geom_pos[geom_id] == pytest.approx(pos)


def test_instinct_limb_and_foot_geometry(model):
  hip_id = _geom(model, "left_hip_collision")
  knee_id = _geom(model, "left_knee_collision")
  foot_id = _geom(model, "left_foot2_collision")
  elbow_id = _geom(model, "left_elbow_collision")
  hand_id = _geom(model, "left_hand_collision")

  assert model.geom_size[hip_id, :2] == pytest.approx((0.05, 0.1))
  assert model.geom_pos[hip_id] == pytest.approx((0.02, 0.0, -0.1))
  assert model.geom_size[knee_id, :2] == pytest.approx((0.05, 0.1))
  assert model.geom_pos[knee_id] == pytest.approx((0.02, 0.0, -0.1))
  assert model.geom_size[foot_id, :2] == pytest.approx((0.008, 0.0835))
  assert model.geom_size[elbow_id, :2] == pytest.approx((0.035, 0.065))
  assert model.geom_pos[elbow_id] == pytest.approx((0.055, 0.0, -0.01))
  assert model.geom_size[hand_id, :2] == pytest.approx((0.05, 0.025))
  assert model.geom_pos[hand_id] == pytest.approx((0.075, 0.0, 0.0))


def _compiled_actuator_table(monkeypatch, effort: str | None):
  """(kp, kd, forcerange, armature) per joint, plus the ctrl->joint order."""
  import importlib

  import gmtrack.assets.unitree_g1 as unitree_g1

  if effort is None:
    monkeypatch.delenv("GMTRACK_HIP_PITCH_EFFORT", raising=False)
  else:
    monkeypatch.setenv("GMTRACK_HIP_PITCH_EFFORT", effort)
  importlib.reload(unitree_g1)

  from mjlab.entity import Entity

  model = Entity(unitree_g1.get_gmtrack_g1_robot_cfg()).spec.compile()
  joints = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).split("/")[-1]
    for i in range(model.njnt)
  ]
  table, order = {}, []
  for i in range(model.nu):
    joint_id = model.actuator_trnid[i, 0]
    name = joints[joint_id]
    order.append(name)
    table[name] = (
      float(model.actuator_gainprm[i, 0]),
      float(-model.actuator_biasprm[i, 2]),
      float(model.actuator_forcerange[i, 1]),
      float(model.dof_armature[model.jnt_dofadr[joint_id]]),
    )
  return table, order


def test_hip_pitch_override_leaves_the_action_scale_alone(monkeypatch):
  """Eq. 3's action scale is a fixed reproduction assumption; the override is physics.

  ``kp * scale == 0.25 * effort``, so deriving the scale from a raised effort limit
  would change what one unit of policy output means and reinterpret every trained
  checkpoint. The env cfg must keep mjlab's frozen dict either way.
  """
  import importlib

  from mjlab.asset_zoo.robots import G1_ACTION_SCALE

  import gmtrack.envs.env_cfg as env_cfg

  monkeypatch.setenv("GMTRACK_HIP_PITCH_EFFORT", "139")
  importlib.reload(env_cfg)
  cfg = env_cfg.make_gmtrack_env_cfg(manifest="unused.json")
  scale = cfg.actions["joint_pos"].scale

  assert scale == G1_ACTION_SCALE
  # Torque per unit of policy output stays 0.25 * 88, not 0.25 * 139.
  kp = 40.17923863450712
  assert scale[".*_hip_pitch_joint"] * kp == pytest.approx(0.25 * 88.0)


def test_default_path_keeps_mjlabs_exact_actuator_layout(monkeypatch):
  """Without an override nothing may be rebuilt -- not even the ctrl ordering.

  ``EntityCfg.sort_actuators`` is False, so giving hip pitch its own actuator entry
  moves it to the end of that block in ctrl order. Per-joint gains survive and mjlab
  addresses actuators by name, but at mjlab's own effort limit the split buys nothing,
  so the default path must not take the risk at all.
  """
  from mjlab.asset_zoo.robots import get_g1_robot_cfg

  import gmtrack.assets.unitree_g1 as unitree_g1

  pristine = get_g1_robot_cfg()
  pristine.spec_fn = unitree_g1.get_instinct_collision_g1_spec
  from mjlab.entity import Entity

  def order(cfg):
    model = Entity(cfg).spec.compile()
    joints = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).split("/")[-1]
      for i in range(model.njnt)
    ]
    return [joints[model.actuator_trnid[i, 0]] for i in range(model.nu)]

  _, ours = _compiled_actuator_table(monkeypatch, None)
  assert ours == order(pristine)


def test_hip_pitch_override_touches_hip_pitch_only(monkeypatch):
  """The override may move hip pitch's effort limit and nothing else."""
  base, base_order = _compiled_actuator_table(monkeypatch, None)
  new, new_order = _compiled_actuator_table(monkeypatch, "139")

  # The split reorders ctrl (documented), but must not add, drop or re-target one.
  assert sorted(base_order) == sorted(new_order)

  changed = {k for k in base if base[k] != new[k]}
  assert changed == {"left_hip_pitch_joint", "right_hip_pitch_joint"}
  for joint in changed:
    kp, kd, _force, armature = base[joint]
    assert new[joint] == (kp, kd, 139.0, armature), "only the effort limit may move"


def test_hip_pitch_effort_override_rejects_nonsense(monkeypatch):
  import importlib

  import gmtrack.assets.unitree_g1 as unitree_g1

  monkeypatch.setenv("GMTRACK_HIP_PITCH_EFFORT", "0")
  importlib.reload(unitree_g1)
  with pytest.raises(ValueError, match="must be positive"):
    unitree_g1.hip_pitch_effort_limit()
