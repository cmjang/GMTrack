"""Extreme-RGMT environment configurations for the Unitree G1 (29 DoF).

Built from scratch rather than by mutating ``mjlab.tasks.tracking.make_tracking_env_cfg``
because the observation layout differs substantially: the paper's actor consumes
histories and a reference *window* (Sec. III-B, IV-A), not the single-frame reference
mjlab's BeyondMimic port uses.

Every numeric constant traceable to the paper is tagged with its table.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from ex_grmt import mdp
from ex_grmt.mdp.commands import MultiMotionCommandCfg

##
# Robot topology.
##

ANCHOR_BODY = "torso_link"

TRACKED_BODIES: tuple[str, ...] = (
  # MUST start with the floating-base root link: MultiMotionCommand writes
  # body_pos_w[:, 0] to the free joint on reset.
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

END_EFFECTORS: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)

FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"

HISTORY_LENGTH = 10
"""``H``: 10-frame proprioceptive and action history (paper Sec. III-B)."""

COMMAND_WINDOW_RADIUS = 10
"""``L``: the reference window has 2L+1 = 21 tokens (paper Sec. III-B)."""

# Table II command perturbations, reused as reference-state-initialisation noise.
VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.52, 0.52),
}


def _proprio_terms() -> dict[str, ObservationTermCfg]:
  """``o^prop = [g_proj(3), omega(3), q - q0(29), qdot(29)]`` -- 64 dims (Eq. 1).

  ORDER IS LOAD-BEARING: ``ExGRMTActorCfg.proprio_term_dims`` must match it, because
  mjlab flattens history per term and the actor splits the flat row back apart
  positionally. Noise magnitudes are Table II's observation-noise column.
  """
  return {
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  """Privileged critic observations (paper Sec. III-B), noise-free."""
  terms: dict[str, ObservationTermCfg] = {
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # Privileged from here on: not available on the real robot.
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "ref_root_height": ObservationTermCfg(
      func=mdp.motion_ref_root_height, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "command_window": ObservationTermCfg(
      func=mdp.motion_command_window, params={"command_name": "motion"}
    ),
  }
  return terms


def make_ex_grmt_env_cfg(
  manifest: str,
  acquisition_clips: str | None = None,
  consolidation_clips: str | None = None,
  acquisition_fraction: float | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the Extreme-RGMT tracking environment.

  Args:
    manifest: Clip manifest from ``ex_grmt.scripts.prepare_motions``.
    acquisition_clips: Challenging set ``D_c``. None = the whole library (Stage I).
    consolidation_clips: Mastered set ``D_m``. Required for Stage II.
    acquisition_fraction: ``xi``. None disables the PACE split (Stage I).
    play: Deterministic replay mode (no corruption, no pushes, no RSI noise).
  """

  ##
  # Observations
  ##

  observations = {
    "proprio_hist": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=True,
      # Flattened per term -> [g(H*3), omega(H*3), q(H*29), qdot(H*29)].
      history_length=HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "action_hist": ObservationGroupCfg(
      terms={"actions": ObservationTermCfg(func=mdp.last_action)},
      concatenate_terms=True,
      enable_corruption=False,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "command_window": ObservationGroupCfg(
      terms={
        "window": ObservationTermCfg(
          func=mdp.motion_command_window,
          params={"command_name": "motion"},
          # Table II command perturbations. Applied uniformly across the window;
          # per-channel magnitudes would need a per-token noise vector, which mjlab's
          # scalar UniformNoiseCfg does not express.
          noise=Unoise(n_min=-0.1, n_max=0.1),
        )
      },
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_critic_terms(),
      concatenate_terms=True,
      enable_corruption=False,
    ),
    # Not a policy input -- see ex_grmt.mdp.observations.motion_star_meta.
    "star": ObservationGroupCfg(
      terms={
        "meta": ObservationTermCfg(
          func=mdp.motion_star_meta, params={"command_name": "motion"}
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MultiMotionCommandCfg(
      entity_name="robot",
      # Episodes end when the clip ends or on termination, never on a timer.
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      manifest=manifest,
      anchor_body_name=ANCHOR_BODY,
      body_names=TRACKED_BODIES,
      acquisition_clips=acquisition_clips,
      consolidation_clips=consolidation_clips,
      acquisition_fraction=acquisition_fraction,
      command_window_radius=COMMAND_WINDOW_RADIUS,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),  # Table II: joint pose +-0.1 rad.
      sampling_mode="adaptive",
    )
  }

  ##
  # Events -- paper Table II, dynamics randomization.
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),  # external push interval [1, 3] s
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "ground_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
        "operation": "abs",
        "ranges": (0.10, 1.75),
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_BODY,)),
        "operation": "add",
        "ranges": (-3.0, 6.0),  # added base mass [-3, 6] kg
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_BODY,)),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),  # x
          1: (-0.05, 0.05),  # y
          2: (-0.05, 0.05),  # z
        },
      },
    ),
    "motor_strength": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "effort_limit_range": (0.8, 1.2),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),  # motor zero offset [-0.01, 0.01] rad
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "ranges": (1.0, 1.05),
      },
    ),
  }

  ##
  # Rewards -- paper Table I.
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-0.1,
      params={"sensor_name": "nonfoot_ground_contact"},
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": END_EFFECTORS,
      },
    ),
  }

  ##
  # Scene
  ##

  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  nonfoot_ground = ContactSensorCfg(
    name="nonfoot_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=(FOOT_GEOM_REGEX,),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_g1_robot_cfg()},
      sensors=(self_collision, feet_ground, nonfoot_ground),
      num_envs=1,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ANCHOR_BODY,
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    # 0.005 s * 4 = 50 Hz policy rate (paper Sec. VI-C). NOTE: this puts the PD
    # controller at 200 Hz, not the paper's 500 Hz; raising it costs 2.5x sim time.
    decimation=4,
    episode_length_s=10.0,
  )

  if play:
    cfg.episode_length_s = int(1e9)
    for group in ("proprio_hist", "command_window"):
      cfg.observations[group].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
