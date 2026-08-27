"""Extreme-RGMT environment configurations for the Unitree G1 (29 DoF).

Built from scratch rather than by mutating ``mjlab.tasks.tracking.make_tracking_env_cfg``
because the observation layout differs substantially: the paper's actor consumes
histories and a reference *window* (Sec. III-B, IV-A), not the single-frame reference
mjlab's BeyondMimic port uses.

Every numeric constant traceable to the paper is tagged with its table.
"""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
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
from mjlab.viewer import ViewerConfig

from ex_grmt import mdp
from ex_grmt.assets import get_ex_grmt_g1_robot_cfg
from ex_grmt.mdp.actions import ReferenceResidualJointPositionActionCfg
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
HAND_GEOM_REGEX = r"^(left|right)_hand_collision$"
"""Wrist-mounted hand capsules (on ``*_wrist_yaw_link``). Exempt from the
undesired-contact penalty: Table I never defines which bodies are "undesired", and
InstinctLab's BeyondMimic shadowing excludes exactly ankles + wrists -- get-ups,
rolls and hand-support motions legitimately put the wrists on the floor, and
penalizing that fights the tracking terms head-on. Elbows/knees stay penalized."""

POLICY_HZ = 50.0
"""Control rate (paper Sec. VI-C). Must match the motion clips' fps."""

DEFAULT_SIM_HZ = 200.0
"""Physics rate. MuJoCo position actuators are evaluated every physics step, so this
is also the simulated PD rate.

The paper's "the low-level PD controller operates at 500 Hz" (Sec. VI-C) describes the
**onboard controller of the real G1**, in the hardware-deployment section; the paper
never states a simulation timestep. 200 Hz with a 50 Hz policy is the standard
sim-side setup and mjlab's own default, so that is what we use. Raise it if you want
tighter contact resolution -- it costs sim time proportionally.
"""

HISTORY_LENGTH = 10
"""``H``: 10-frame proprioceptive and action history (paper Sec. III-B)."""

COMMAND_WINDOW_RADIUS = 10
"""``L``: the reference window has 2L+1 = 21 tokens (paper Sec. III-B)."""

CAUSAL_ACTOR_WINDOW_OFFSETS: tuple[int, ...] = (
  -32,
  -24,
  -16,
  -12,
  -8,
  -6,
  -4,
  -3,
  -2,
  -1,
  0,
)
"""Near-dense/far-sparse causal history spanning 0.64 s at 50 Hz.

DAJI (arXiv:2605.14417) and TeleGate (arXiv:2602.09628) both use dense recent
samples and sparse older samples. This configurable local layout extends that idea
to a longer window after the actor's positional encoding was made offset-aware.
"""

CAUSAL_CRITIC_WINDOW_OFFSETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
"""Privileged future coverage through 1.28 s for the training-only critic."""

CAUSAL_RECONSTRUCTION_OFFSETS: tuple[int, ...] = (5, 10, 20)
"""Sparse future-prediction targets from TeleGate (arXiv:2602.09628)."""

# Table II, command perturbation -- per *channel*, not a single magnitude:
#   base linear velocity  +-0.5  m/s
#   base angular velocity +-0.52 rad/s
#   gravity direction      0.05
#   joint pose            +-0.1  rad
# One token is [v_ref(3), w_ref(3), g_ref(3), q_ref(29)] = 38 channels; the window is
# 21 such tokens flattened. mjlab's UniformNoiseCfg accepts a per-dimension tuple, so
# the pattern is tiled to the full flat width rather than approximated by a scalar
# (a single +-0.1 would under-perturb velocities 5x and over-perturb gravity 2x).
_COMMAND_TOKEN_NOISE: tuple[float, ...] = (
  (0.5,) * 3 + (0.52,) * 3 + (0.05,) * 3 + (0.1,) * 29
)


def command_window_noise(
  heading_closed_loop: bool = False,
  *,
  num_window_tokens: int = 2 * COMMAND_WINDOW_RADIUS + 1,
) -> tuple[float, ...]:
  """Per-channel command-window noise for the 38D or heading-aware 44D token.

  The first 38 magnitudes are the unchanged Extreme-RGMT Table-II values. The
  opt-in six-dimensional relative orientation follows SONIC's 0.05 magnitude.
  """
  token = _COMMAND_TOKEN_NOISE + ((0.05,) * 6 if heading_closed_loop else ())
  return token * num_window_tokens


# Public legacy constant: keep the faithful default exactly 21 x 38 channels.
COMMAND_WINDOW_NOISE: tuple[float, ...] = command_window_noise()

# Push / reference-state-initialisation velocity noise. Table II specifies only the
# push *interval*; the magnitudes below follow InstinctLab's BeyondMimic shadowing
# (identical to its push and RSI ranges). x/y/roll/pitch coincide with Table II's
# command perturbations; z and yaw have no paper value.
VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

# Fall recovery -- RGMT (arXiv:2601.23080v1) Sec. II-D. Extreme-RGMT builds on RGMT's
# controller/training design and demonstrates landing recovery on hardware, but never
# restates the mechanism. Recovery poses are generated at runtime and do not require
# fall/get-up demonstrations or a separate motion corpus.
RECOVERY_PROBABILITY = 0.15
"""A resetting env becomes a recovery env with this probability (RGMT Sec. II-D)."""
RECOVERY_WINDOW_S = 3.0
"""Instability terminations are suspended for this long (RGMT Sec. II-D)."""
RECOVERY_ASSIST_FORCE_N = (0.0, 200.0)
"""Upward assistance-force magnitude range, newtons (RGMT Sec. II-D)."""
RECOVERY_ASSIST_ANNEAL_STEPS = 2_400_000
"""Linear anneal horizon in env steps (= 100k iterations x 24 rollout steps). RGMT
gives no number ("annealed over training iterations"), so this tracks the Stage-I run
length in ``rl_cfgs.stage1_runner_cfg``; change both together. See
MultiMotionCommandCfg.

Counted from the first step taken *with recovery enabled*, not from the start of
training history -- ``MultiMotionCommand.recovery_steps_elapsed``."""
RECOVERY_ROOT_HEIGHT_M = (0.35, 0.65)
"""ASSUMPTION: randomized recovery-pose pelvis height; RGMT gives no distribution."""
RECOVERY_ROOT_TILT_RAD = (math.pi / 3.0, 2.0 * math.pi / 3.0)
"""ASSUMPTION: non-upright tilt range around a random horizontal axis."""
RECOVERY_JOINT_JITTER_RAD = (-0.25, 0.25)
"""ASSUMPTION: per-joint jitter around the ordinary reference, clipped to limits."""


def _proprio_terms(
  acquisition_fraction: float | None, noise_enabled: bool
) -> dict[str, ObservationTermCfg]:
  """``o^prop = [g_proj(3), omega(3), q - q0(29), qdot(29)]`` -- 64 dims (Eq. 1).

  ORDER IS LOAD-BEARING: ``ExGRMTActorCfg.proprio_term_dims`` must match it, because
  mjlab flattens history per term and the actor splits the flat row back apart
  positionally. Noise magnitudes are Table II's observation-noise column.
  """
  return {
    "projected_gravity": ObservationTermCfg(
      func=mdp.role_noisy_projected_gravity,
      params={
        "acquisition_fraction": acquisition_fraction,
        "magnitude": 0.05,
        "enabled": noise_enabled,
      },
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.role_noisy_builtin_sensor,
      params={
        "sensor_name": "robot/imu_ang_vel",
        "acquisition_fraction": acquisition_fraction,
        "magnitude": 0.2,
        "enabled": noise_enabled,
      },
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.role_noisy_joint_pos_rel,
      params={
        "biased": True,
        "acquisition_fraction": acquisition_fraction,
        "magnitude": 0.01,
        "enabled": noise_enabled,
      },
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.role_noisy_joint_vel_rel,
      params={
        "acquisition_fraction": acquisition_fraction,
        "magnitude": 0.5,
        "enabled": noise_enabled,
      },
    ),
  }


def _critic_terms() -> dict[str, ObservationTermCfg]:
  """Critic input ``s_t = [o_t, g_t, o_t^priv]`` (RGMT Eq. 3-4), noise-free.

  RGMT Eq. 1 includes the previous action inside ``o_t``; ``g_t`` is the *single*
  current reference token (Eq. 2), not the actor's 21-token window; the privileged
  block is exactly ``[h_t^ref, x_t^link, v_t]``. Extreme-RGMT Sec. III-B repeats the
  same categories. An earlier revision additionally fed the critic the full command
  window plus reference joint targets -- more information than the paper's critic,
  which shifts value estimates and hence GAE/STAR advantage statistics.
  """
  return {
    # o_t (noise-free): [g_proj, omega, q - q0, qdot, a_{t-1}] (RGMT Eq. 1).
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # g_t: current reference token [v_ref, w_ref, g_ref, q_ref] (RGMT Eq. 2).
    "command_token": ObservationTermCfg(
      func=mdp.motion_command_token, params={"command_name": "motion"}
    ),
    # o_t^priv = [h_t^ref, x_t^link, v_t] (RGMT Eq. 3): not available on-robot.
    "ref_root_height": ObservationTermCfg(
      func=mdp.motion_ref_root_height, params={"command_name": "motion"}
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
  }


def make_ex_grmt_env_cfg(
  manifest: str,
  acquisition_clips: str | None = None,
  consolidation_clips: str | None = None,
  acquisition_fraction: float | None = None,
  play: bool = False,
  sim_hz: float = DEFAULT_SIM_HZ,
  experimental_rsi: bool = False,
  heading_closed_loop: bool = False,
  recovery_probability: float = 0.0,
  require_v1_stratification: bool = False,
  stratification_mastered_manifest: str | None = None,
  stratification_challenging_manifest: str | None = None,
  causal_online: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the Extreme-RGMT tracking environment.

  Args:
    manifest: Complete-sequence Stage-I manifest or post-Stage-I logical-clip manifest.
    acquisition_clips: Challenging set ``D_c``. None = the whole library (Stage I).
    consolidation_clips: Mastered set ``D_m``. Required for Stage II.
    acquisition_fraction: ``xi``. None disables the PACE split (Stage I).
    play: Deterministic replay mode (no corruption, no pushes, no RSI noise).
    sim_hz: Physics rate (= simulated PD rate). See :data:`DEFAULT_SIM_HZ`.
    experimental_rsi: Enable the pose/velocity/joint reset jitter inherited from
      mjlab. It is not part of Extreme-RGMT's published perturbation protocol and is
      disabled in the faithful configuration.
    heading_closed_loop: Append SONIC-style relative root orientation to each command
      token. Training resets receive yaw-only jitter of +/-0.2 rad so the feedback is
      exercised; play remains deterministic. False preserves the 38D baseline.
    recovery_probability: Opt-in RGMT recovery mechanism. The randomized-pose
      distribution and anneal horizon are explicit local assumptions because RGMT
      does not publish those parameters.
    require_v1_stratification: Validate Stage-II manifests against the v1
      10-second/5-rollout/80-percent protocol when the command is constructed.
    stratification_mastered_manifest: Authenticated D_m manifest used for strict
      validation independently of which subset this task samples.
    stratification_challenging_manifest: Authenticated D_c manifest used for strict
      validation independently of which subset this task samples.
    causal_online: Use the configurable strictly causal actor window, a training-only
      stochastic intent reconstruction objective, and a history-conditioned
      privileged critic with a masked future reference window.
  """
  if abs(sim_hz / POLICY_HZ - round(sim_hz / POLICY_HZ)) > 1e-9:
    raise ValueError(
      f"sim_hz={sim_hz} is not an integer multiple of the {POLICY_HZ} Hz policy rate."
    )

  actor_window_offsets = CAUSAL_ACTOR_WINDOW_OFFSETS if causal_online else None
  num_actor_window_tokens = (
    len(actor_window_offsets)
    if actor_window_offsets is not None
    else 2 * COMMAND_WINDOW_RADIUS + 1
  )

  ##
  # Observations
  ##

  observations = {
    "proprio_hist": ObservationGroupCfg(
      terms=_proprio_terms(acquisition_fraction, noise_enabled=not play),
      concatenate_terms=True,
      # Noise is applied inside role-aware observation terms, because mjlab's group
      # corruption has no environment-role mask.
      enable_corruption=False,
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
          func=mdp.role_noisy_motion_command_window,
          params={
            "command_name": "motion",
            "acquisition_fraction": acquisition_fraction,
            "magnitude": command_window_noise(
              heading_closed_loop, num_window_tokens=num_actor_window_tokens
            ),
            "enabled": not play,
          },
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
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
  if causal_online:
    observations["history_valid_mask"] = ObservationGroupCfg(
      terms={
        "mask": ObservationTermCfg(
          func=mdp.executed_history_valid_mask,
          params={"history_length": HISTORY_LENGTH},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    observations["past_valid_mask"] = ObservationGroupCfg(
      terms={
        "mask": ObservationTermCfg(
          func=mdp.motion_command_past_valid_mask,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    observations["command_future_window"] = ObservationGroupCfg(
      terms={
        "window": ObservationTermCfg(
          func=mdp.motion_command_future_window,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    observations["future_valid_mask"] = ObservationGroupCfg(
      terms={
        "mask": ObservationTermCfg(
          func=mdp.motion_command_future_valid_mask,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    # Training-only target groups are kept outside actor/critic obs_groups. They ride
    # in rollout storage so PacePPO can apply reconstruction/KL to the exact
    # acquisition rows selected by its existing pace_env_split-derived mask.
    observations["future_reconstruction_target"] = ObservationGroupCfg(
      terms={
        "target": ObservationTermCfg(
          func=mdp.motion_command_reconstruction_target,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    observations["future_reconstruction_valid_mask"] = ObservationGroupCfg(
      terms={
        "mask": ObservationTermCfg(
          func=mdp.motion_command_reconstruction_valid_mask,
          params={"command_name": "motion"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    )

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    # Paper Eq. (3): the residual is added to the *reference* joint pose, not to the
    # constant default pose that mjlab's JointPositionAction uses. See
    # ex_grmt.mdp.actions for why the difference matters.
    "joint_pos": ReferenceResidualJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      # ASSUMPTION: Eq. (3) writes q_ref + a_t with no scale factor. We keep mjlab's
      # per-joint scale (0.25 * effort_limit / stiffness) so the policy's unit-ish
      # Gaussian output maps to a sensible joint range; Table III lists no action
      # scale, so the paper must apply an equivalent normalization implicitly.
      #
      # Deliberately mjlab's frozen dict, NOT a value derived from our own effort
      # limits. `hip_pitch_effort_limit()` can raise the hip-pitch torque clamp, and
      # since `kp * scale == 0.25 * effort`, re-deriving the scale would also change
      # what a unit of policy output means -- silently reinterpreting every trained
      # checkpoint. The scale stays pinned; the override moves the clamp only.
      scale=G1_ACTION_SCALE,
      command_name="motion",
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
      require_v1_stratification=require_v1_stratification,
      stratification_mastered_manifest=(
        stratification_mastered_manifest or consolidation_clips
      ),
      stratification_challenging_manifest=(
        stratification_challenging_manifest or acquisition_clips
      ),
      command_window_radius=COMMAND_WINDOW_RADIUS,
      command_window_offsets=actor_window_offsets,
      require_causal_window=causal_online,
      critic_window_offsets=(
        CAUSAL_CRITIC_WINDOW_OFFSETS if causal_online else None
      ),
      reconstruction_window_offsets=(
        CAUSAL_RECONSTRUCTION_OFFSETS if causal_online else None
      ),
      heading_closed_loop=heading_closed_loop,
      # These affect the simulated initial state, not the command observation.  The
      # paper's +/-0.1 rad entry belongs to command perturbation, already applied
      # above, so RSI stays off unless an explicit ablation requests it.
      pose_range=(
        {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (-0.01, 0.01),
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-0.2, 0.2),
        }
        if experimental_rsi
        else ({"yaw": (-0.2, 0.2)} if heading_closed_loop and not play else {})
      ),
      velocity_range=VELOCITY_RANGE if experimental_rsi else {},
      joint_position_range=(-0.1, 0.1) if experimental_rsi else (0.0, 0.0),
      sampling_mode="adaptive",
      # Recovery comes from RGMT rather than a fully specified Extreme-RGMT-v1
      # protocol. Explicit recovery tasks opt in with RECOVERY_PROBABILITY.
      recovery_probability=recovery_probability,
      recovery_window_s=RECOVERY_WINDOW_S,
      recovery_assist_force_range=RECOVERY_ASSIST_FORCE_N,
      recovery_assist_anneal_steps=RECOVERY_ASSIST_ANNEAL_STEPS,
      recovery_root_height_range=RECOVERY_ROOT_HEIGHT_M,
      recovery_root_tilt_range=RECOVERY_ROOT_TILT_RAD,
      recovery_joint_position_jitter=RECOVERY_JOINT_JITTER_RAD,
    )
  }

  ##
  # Events -- paper Table II, dynamics randomization.
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.role_push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),  # external push interval [1, 3] s
      params={
        "acquisition_fraction": acquisition_fraction,
        "velocity_range": VELOCITY_RANGE,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # RGMT Sec. II-D upward assistance for recovery environments. Step mode: the
    # wrench must be rewritten (and zeroed) every step because xfrc_applied persists
    # across resets.
    "recovery_assist": EventTermCfg(
      func=mdp.recovery_assist_force,
      mode="step",
      params={"command_name": "motion", "asset_cfg": SceneEntityCfg("robot")},
    ),
    # MuJoCo takes max(friction_terrain, friction_robot) for equal-priority contacts.
    # Pinning the acquisition plane to Table II's lower bound makes the shared
    # per-environment robot draw below effective instead of clipping every sample
    # below 1.0. Consolidation keeps the nominal plane coefficient of 1.0.
    "terrain_friction_floor": EventTermCfg(
      mode="startup",
      func=mdp.role_geom_friction,
      params={
        "acquisition_fraction": acquisition_fraction,
        "asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)),
        "operation": "abs",
        "ranges": (0.10, 0.10),
        "shared_random": True,
      },
    ),
    "ground_friction": EventTermCfg(
      mode="startup",
      func=mdp.role_geom_friction,
      params={
        "acquisition_fraction": acquisition_fraction,
        # Whole-body motions contact the ground with hands, knees and torso too.
        "asset_cfg": SceneEntityCfg("robot", geom_names=r".*_collision\d*$"),
        "operation": "abs",
        "ranges": (0.10, 1.75),
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=mdp.role_body_mass,
      params={
        "acquisition_fraction": acquisition_fraction,
        "asset_cfg": SceneEntityCfg("robot", body_names=(ANCHOR_BODY,)),
        "operation": "add",
        "ranges": (-3.0, 6.0),  # added base mass [-3, 6] kg
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.role_body_com_offset,
      params={
        "acquisition_fraction": acquisition_fraction,
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
      func=mdp.role_effort_limits,
      params={
        "acquisition_fraction": acquisition_fraction,
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "effort_limit_range": (0.8, 1.2),
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=mdp.role_pd_gains,
      params={
        "acquisition_fraction": acquisition_fraction,
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.role_encoder_bias,
      params={
        "acquisition_fraction": acquisition_fraction,
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),  # motor zero offset [-0.01, 0.01] rad
      },
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=mdp.role_joint_armature,
      params={
        "acquisition_fraction": acquisition_fraction,
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "ranges": (1.0, 1.05),
      },
    ),
  }

  if recovery_probability <= 0.0:
    # Do not execute an unpublished assistance event on the strict-v1 path. The
    # recovery-aware termination wrappers are no-ops when no episode is marked.
    events.pop("recovery_assist")

  ##
  # Rewards -- paper Table I.
  ##

  rewards: dict[str, RewardTermCfg] = {
    # NOTE: Table I lists five tracking terms and does NOT include a global anchor
    # *position* reward -- only orientation. mjlab's BeyondMimic port does include
    # one (weight 0.5); we drop it to match the paper. The design is coherent without
    # it: body positions are tracked relative to the anchor, global body velocities
    # are tracked in world frame, and absolute XY drift is intentionally unpenalized
    # because a retargeted clip's world position is arbitrary. Height is still
    # constrained, by the `anchor_pos` termination.
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
    # The threshold-based tracking checks below cannot detect NaN because every
    # comparison with NaN is false.  Reset a corrupt MuJoCo world before its state is
    # sensed and appended to proprio_hist.
    "nonfinite_physics_state": TerminationTermCfg(func=mdp.nonfinite_physics_state),
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "motion_sequence_end": TerminationTermCfg(
      func=mdp.motion_sequence_end,
      time_out=True,
      params={"command_name": "motion"},
    ),
    # RGMT Sec. II-D's three instability conditions -- "excessive base orientation
    # deviation, insufficient base height, and abnormally low height of key body
    # links" -- as symmetric reference deviations, identical to mjlab's BeyondMimic
    # port, which both papers state they follow. All three wear the recovery shield:
    # suspended for a recovery environment's first RECOVERY_WINDOW_S seconds,
    # unchanged everywhere else.
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only_outside_recovery,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori_outside_recovery,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only_outside_recovery,
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
      # Feet and wrists may touch the ground unpenalized (see HAND_GEOM_REGEX).
      exclude=(FOOT_GEOM_REGEX, HAND_GEOM_REGEX),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_ex_grmt_g1_robot_cfg()},
      sensors=(self_collision, feet_ground, nonfoot_ground),
      # Per-GPU environment count -- mjlab's distributed launcher gives every rank
      # this many, so 4 GPUs is 4096 total (BeyondMimic/SONIC both train at 4096).
      # Training jobs no longer pass --env.scene.num-envs; play/tests override it.
      num_envs=1024,
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
      # The InstinctMJ whole-body collision profile can create substantially more
      # constraint rows than mjlab's stock G1, especially during randomized
      # recovery resets.  The stock njmax=250 overflowed in long cluster runs and
      # the truncated solve subsequently produced NaN proprioception.  Observed
      # peaks were 418 rows, so keep headroom for unseen contact configurations.
      njmax=512,
      # Three contact sensors may match the same dense fallen-body contact set.
      # MuJoCo Warp reported peaks of 74 with its default allocation of 64.
      contact_sensor_maxmatch=128,
      mujoco=MujocoCfg(timestep=1.0 / sim_hz, iterations=10, ls_iterations=20),
    ),
    # 50 Hz policy (paper Sec. VI-C) over a 200 Hz sim. Derived so the two can never
    # drift apart: the policy rate must equal the motion clips' fps, or the reference
    # advances at a different rate than the controller.
    decimation=int(round(sim_hz / POLICY_HZ)),
    # Training rollout horizon. Stage I may load longer complete sequences: adaptive
    # sampling initializes throughout all of their temporal bins, so this does not
    # truncate the stored data. Evaluation disables the timer in scripts/_harness.py.
    episode_length_s=10.0,
  )

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    # Nominal evaluation means nominal dynamics as well as clean observations.
    for event in tuple(cfg.events):
      if event != "recovery_assist":
        cfg.events.pop(event)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    motion_cmd.recovery_probability = 0.0

  return cfg
