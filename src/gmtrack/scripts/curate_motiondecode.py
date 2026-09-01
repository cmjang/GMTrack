"""Build auditable MotionDecode selections for GMTrack.

The recommended ``basic-backflip-3h`` profile is deliberately narrow: it keeps a
roughly paper-sized mix of ordinary flat-ground locomotion, transitions, idle/
posture motion, simple jumps, and every pure ``Back_Somersault`` take.  Continuous
``Forward_Somersault_with_Backward_Flip`` takes remain excluded because they are
terrain/environment dependent.  All terrain-, prop-, object-, multi-person-,
dance-, combat-, and other parkour motion is excluded by default.  The raw dataset
is never modified.

MotionDecode's released G1 CSVs are 120 Hz, Z-up, metres/radians, with a wxyz root
quaternion.  The output is a selection JSON consumed by ``prepare_motions``; it is
not itself a training manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal

import numpy as np

from gmtrack.scripts.prepare_motions import (
  MOTIONDECODE_HEADER,
  _read_source_motion,
)

Profile = Literal["basic-backflip-3h", "flat-no-props", "all-g1"]
_SCHEMA_VERSION = 2
_FPS = 120.0


@dataclass(frozen=True)
class ExclusionRule:
  reason: str
  prefix: str


@dataclass(frozen=True)
class BasicGroup:
  name: str
  prefixes: tuple[str, ...]
  target_seconds: float | None
  difficulty_exception: bool = False


_PARKOUR = "samples/2.3.Extreme_Environment_Interaction/2.3.1.Parkour_Actions"
_BACK_SOMERSAULT = f"{_PARKOUR}/2.3.1.12.Back_Somersault"

FLAT_NO_PROPS_RULES: tuple[ExclusionRule, ...] = (
  ExclusionRule(
    "stairs_steps",
    "samples/1.7.Daily_Life_Behaviors/1.7.3.Locomotion_and_Posture_Change/1.7.3.3.Stair_Climbing_and_Descending",
  ),
  ExclusionRule(
    "stairs_steps",
    "samples/2.1.Basic_Environment_Interaction/2.1.1.Stair_and_Step_Climbing_Descending",
  ),
  ExclusionRule(
    "supporting_furniture",
    "samples/2.1.Basic_Environment_Interaction/2.1.2.Various_Sitting_and_Leaning_Postures",
  ),
  ExclusionRule(
    "obstacle_or_climb",
    "samples/1.2.State_Transition_Category/1.2.3.Action_Transition(Walk_Jump_Climb_Walk)",
  ),
  ExclusionRule(
    "obstacle_or_climb",
    "samples/1.3.Basic_Gait_Category/1.3.4.Jumping_Movement/1.3.4.2.Leaping_over",
  ),
  ExclusionRule(
    "obstacle_or_climb",
    "samples/1.11.Special_Population_Actions/1.11.2.Disability_Assistance_Actions/1.11.2.3.Assisted_Obstacle_Crossing",
  ),
  ExclusionRule(
    "obstacle_or_climb",
    "samples/2.2.Complex_Terrain_Interaction/2.2.1.Obstacle_Crossing",
  ),
  ExclusionRule(
    "uneven_ground",
    "samples/2.2.Complex_Terrain_Interaction/2.2.2.Uneven_Ground_Walking",
  ),
  *(
    ExclusionRule("parkour_prop_or_height", f"{_PARKOUR}/{leaf}")
    for leaf in (
      "2.3.1.1.Precision_Jump",
      "2.3.1.2.Obstacle_Leap",
      "2.3.1.3.Speed_Crossing",
      "2.3.1.4.Lazy_Flip",
      "2.3.1.5.Two_Hand_Vault",
      "2.3.1.6.Duck_Under",
      "2.3.1.8.Run_Up_Wall",
      "2.3.1.9.Balance_Walking",
      "2.3.1.10.Descent_Speed_Control",
      "2.3.1.14.Wall_Flip",
    )
  ),
  ExclusionRule("parkour_ambiguous", f"{_PARKOUR}/2.3.1.20.Others"),
)


BASIC_GROUPS: tuple[BasicGroup, ...] = (
  BasicGroup(
    "normal_walk",
    ("samples/1.3.Basic_Gait_Category/1.3.1.Normal_Walking",),
    4620.0,
  ),
  BasicGroup(
    "brisk_walk",
    (
      "samples/1.3.Basic_Gait_Category/1.3.2.Fast_Walking_Jogging/1.3.2.1.Brisk_walking",
    ),
    1000.0,
  ),
  BasicGroup(
    "jogging",
    ("samples/1.3.Basic_Gait_Category/1.3.2.Fast_Walking_Jogging/1.3.2.2.Jogging",),
    1000.0,
  ),
  BasicGroup(
    "moderate_run",
    (
      "samples/1.3.Basic_Gait_Category/1.3.3.Medium_Speed_Running_Sprinting/1.3.3.1.Moderate_speed_running",
    ),
    600.0,
  ),
  BasicGroup(
    "speed_transition",
    (
      "samples/1.2.State_Transition_Category/1.2.1.Speed_Transition(Still_Walk_Run_Stop)",
    ),
    900.0,
  ),
  BasicGroup(
    "direction_transition",
    (
      "samples/1.2.State_Transition_Category/1.2.2.Direction_Transition(Straight_Turn_Straight)",
    ),
    900.0,
  ),
  BasicGroup(
    "directional_walk",
    (
      "samples/1.4.Constrained_Gait_Category/1.4.6.Lateral_Walking",
      "samples/1.4.Constrained_Gait_Category/1.4.7.Forward_Walking",
      "samples/1.4.Constrained_Gait_Category/1.4.8.Turning_in_Place",
    ),
    600.0,
  ),
  BasicGroup(
    "standing_balance",
    (
      "samples/1.10.Emotional_Expressions_and_Social_Behaviors/1.10.5.Neutral_Postures_Static/1.10.5.1.Standing_Waiting",
      "samples/1.10.Emotional_Expressions_and_Social_Behaviors/1.10.5.Neutral_Postures_Static/1.10.5.5.Slight_Weight_Shift",
    ),
    400.0,
  ),
  BasicGroup(
    "squat_stretch",
    (
      "samples/1.7.Daily_Life_Behaviors/1.7.3.Locomotion_and_Posture_Change/1.7.3.8.Bending_Squatting_Stretching",
    ),
    400.0,
  ),
  BasicGroup(
    "simple_jump",
    (
      "samples/1.1.Basic_Movement_Category/1.1.1.High_Dynamic_Movement/1.1.1.1.Standing_High_Jump",
      "samples/1.3.Basic_Gait_Category/1.3.4.Jumping_Movement/1.3.4.1.Standing_long_jump",
    ),
    600.0,
  ),
  BasicGroup(
    "back_somersault",
    (_BACK_SOMERSAULT,),
    None,
    difficulty_exception=True,
  ),
)

_BACKFLIP_TOKEN = re.compile(
  r"(?:^|_)(?:back_somersault|backflip|back_flip|backward_flip|backward_somersault)(?:_|$)"
)
_METRIC_NAMES = (
  "joint_speed_p99_rad_s",
  "joint_acc_p99_rad_s2",
  "root_lin_speed_p99_m_s",
  "root_ang_speed_p99_rad_s",
  "root_tilt_p99_deg",
)
_BASIC_LIMITS = np.array([12.0, 400.0, 6.0, 8.0, 65.0], dtype=np.float64)


def _normalise_relative(value: str) -> str:
  relative = PurePosixPath(value.replace("\\", "/"))
  if relative.is_absolute() or ".." in relative.parts:
    raise ValueError(f"Expected a safe path relative to input_dir, got {value!r}.")
  return relative.as_posix().rstrip("/")


def _matches_prefix(path: str, prefix: str) -> bool:
  path = _normalise_relative(path)
  prefix = _normalise_relative(prefix)
  return path == prefix or path.startswith(f"{prefix}/")


def _backflip_candidate(path: str) -> bool:
  normalised = re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")
  return _BACKFLIP_TOKEN.search(normalised) is not None


def _has_motiondecode_header(path: Path) -> bool:
  try:
    with path.open() as f:
      header = tuple(f.readline().rstrip("\r\n").split(","))
  except (OSError, UnicodeError):
    return False
  return header == MOTIONDECODE_HEADER


def _rules(
  profile: Profile, extra_prefixes: Iterable[str]
) -> tuple[ExclusionRule, ...]:
  rules = list(FLAT_NO_PROPS_RULES if profile == "flat-no-props" else ())
  rules.extend(
    ExclusionRule("custom", _normalise_relative(prefix)) for prefix in extra_prefixes
  )
  seen: set[str] = set()
  for rule in rules:
    if rule.prefix in seen:
      raise ValueError(f"Duplicate exclusion prefix: {rule.prefix}")
    seen.add(rule.prefix)
  return tuple(rules)


def build_selection(
  input_dir: str | Path,
  *,
  profile: Literal["flat-no-props", "all-g1"] = "flat-no-props",
  extra_prefixes: Iterable[str] = (),
  validate_headers: bool = False,
) -> dict[str, Any]:
  """Return the legacy taxonomy selection without modifying the dataset."""
  root = Path(input_dir).resolve()
  samples_dir = root / "samples"
  if not samples_dir.is_dir():
    raise FileNotFoundError(f"MotionDecode samples directory not found: {samples_dir}")
  rules = _rules(profile, extra_prefixes)
  selected: list[str] = []
  excluded: list[dict[str, str]] = []
  backflips: list[str] = []
  counts: Counter[str] = Counter()
  for file in sorted(samples_dir.rglob("*.csv")):
    relative = file.relative_to(root).as_posix()
    matched = next(
      (rule for rule in rules if _matches_prefix(relative, rule.prefix)), None
    )
    if matched is not None:
      excluded.append({"path": relative, "reason": matched.reason})
      counts[matched.reason] += 1
      continue
    if validate_headers and not _has_motiondecode_header(file):
      excluded.append({"path": relative, "reason": "incompatible_schema"})
      counts["incompatible_schema"] += 1
      continue
    selected.append(relative)
    if _backflip_candidate(relative):
      backflips.append(relative)
  if not selected:
    raise ValueError(f"Curation selected no CSV files under {root}")
  return {
    "kind": "motiondecode_selection",
    "schema_version": _SCHEMA_VERSION,
    "profile": profile,
    "input_dir": str(root),
    "source_format": _source_format(),
    "header_validation": "all_selected"
    if validate_headers
    else "deferred_to_converter",
    "selected_count": len(selected),
    "excluded_count": len(excluded),
    "excluded_counts_by_reason": dict(sorted(counts.items())),
    "backflip_candidate_count": len(backflips),
    "backflip_candidates": backflips,
    "exclusion_rules": [rule.__dict__ for rule in rules],
    "selected_files": selected,
    "excluded_files": excluded,
  }


def _source_format() -> dict[str, Any]:
  return {
    "fps": int(_FPS),
    "root_position": "xyz_metres_z_up",
    "root_quaternion": "wxyz",
    "joint_count": 29,
  }


def _kinematic_metrics(
  motion: np.ndarray, *, root_height_max: float = 2.0
) -> np.ndarray:
  joint_pos = motion[:, 7:]
  root_pos = motion[:, :3]
  root_quat = motion[:, 3:7]
  quat_norm = np.linalg.norm(root_quat, axis=1)
  if not np.allclose(quat_norm, 1.0, atol=2.0e-3):
    raise ValueError("root quaternion is not unit length")
  if np.max(np.abs(joint_pos)) > 3.2:
    raise ValueError("joint position exceeds the G1 curation bound")
  if root_pos[:, 2].min() < 0.1 or root_pos[:, 2].max() > root_height_max:
    raise ValueError(f"root height is outside [0.1, {root_height_max:.1f}] m")

  joint_vel = np.diff(joint_pos, axis=0) * _FPS
  joint_acc = np.diff(joint_vel, axis=0) * _FPS
  root_lin_speed = np.linalg.norm(np.diff(root_pos, axis=0) * _FPS, axis=1)
  dots = np.abs(np.sum(root_quat[:-1] * root_quat[1:], axis=1))
  root_ang_speed = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0)) * _FPS
  root_z_alignment = np.clip(
    1.0 - 2.0 * (np.square(root_quat[:, 1]) + np.square(root_quat[:, 2])),
    -1.0,
    1.0,
  )
  root_tilt = np.rad2deg(np.arccos(root_z_alignment))
  if joint_acc.size == 0:
    raise ValueError("motion is too short for acceleration validation")
  return np.array(
    [
      np.quantile(np.abs(joint_vel), 0.99),
      np.quantile(np.abs(joint_acc), 0.99),
      np.quantile(root_lin_speed, 0.99),
      np.quantile(root_ang_speed, 0.99),
      np.quantile(root_tilt, 0.99),
    ],
    dtype=np.float64,
  )


def _candidate_order(root: Path, paths: Iterable[Path]) -> list[Path]:
  def key(path: Path) -> bytes:
    relative = path.relative_to(root).as_posix().encode()
    return hashlib.sha256(relative).digest()

  return sorted(paths, key=key)


def _outside_reason(relative: str) -> str:
  if _matches_prefix(relative, "samples/2.1.Basic_Environment_Interaction"):
    return "environment_or_support_dependency"
  if _matches_prefix(relative, "samples/2.2.Complex_Terrain_Interaction"):
    return "terrain_dependency"
  if _matches_prefix(relative, "samples/2.3.Extreme_Environment_Interaction"):
    return "other_parkour_or_environment"
  if any(
    _matches_prefix(relative, prefix)
    for prefix in (
      "samples/1.13.Object_Handling_and_Operation",
      "samples/2.4.Competitive_Interaction",
      "samples/3.1.Object_Handling",
      "samples/3.2.Environment_Operation",
      "samples/3.3.Ball_Game_Interaction",
    )
  ):
    return "object_or_interaction_dependency"
  if any(
    _matches_prefix(relative, prefix)
    for prefix in (
      "samples/1.1.Basic_Movement_Category/1.1.2.Ground_Recovery_Movement",
      "samples/1.5.Balance_Control_Actions",
      "samples/1.6.Crawling_Category",
      "samples/1.14.Dance_and_Performance",
      "samples/4.Martial_Arts",
      "samples/5.Dance",
    )
  ):
    return "hard_or_nonbasic_motion"
  return "outside_basic_allowlist"


def build_basic_backflip_selection(input_dir: str | Path) -> dict[str, Any]:
  root = Path(input_dir).resolve()
  samples_dir = root / "samples"
  if not samples_dir.is_dir():
    raise FileNotFoundError(f"MotionDecode samples directory not found: {samples_dir}")
  all_files = sorted(samples_dir.rglob("*.csv"))
  by_group: dict[str, list[Path]] = {group.name: [] for group in BASIC_GROUPS}
  group_for_relative: dict[str, BasicGroup] = {}
  for path in all_files:
    relative = path.relative_to(root).as_posix()
    matches = [
      group
      for group in BASIC_GROUPS
      if any(_matches_prefix(relative, prefix) for prefix in group.prefixes)
    ]
    if len(matches) > 1:
      raise RuntimeError(f"Basic allowlist groups overlap for {relative}: {matches}")
    if matches:
      group = matches[0]
      by_group[group.name].append(path)
      group_for_relative[relative] = group

  selected_records: list[dict[str, Any]] = []
  selected_paths: set[str] = set()
  candidate_exclusions: dict[str, str] = {}
  group_summary: dict[str, dict[str, Any]] = {}
  for group in BASIC_GROUPS:
    accepted_seconds = 0.0
    accepted = 0
    rejected = 0
    candidates = _candidate_order(root, by_group[group.name])
    for path in candidates:
      relative = path.relative_to(root).as_posix()
      if group.target_seconds is not None and accepted_seconds >= group.target_seconds:
        candidate_exclusions[relative] = "family_duration_quota"
        continue
      try:
        motion = _read_source_motion(path, "motiondecode")
        metrics = _kinematic_metrics(
          motion, root_height_max=2.5 if group.difficulty_exception else 2.0
        )
      except ValueError as exc:
        candidate_exclusions[relative] = f"invalid_motion:{exc}"
        rejected += 1
        continue
      if not group.difficulty_exception and np.any(metrics > _BASIC_LIMITS):
        failed = [
          name
          for name, value, limit in zip(
            _METRIC_NAMES, metrics, _BASIC_LIMITS, strict=True
          )
          if value > limit
        ]
        candidate_exclusions[relative] = f"kinematic_limit:{','.join(failed)}"
        rejected += 1
        continue
      duration = (motion.shape[0] - 1) / _FPS
      selected_paths.add(relative)
      accepted_seconds += duration
      accepted += 1
      selected_records.append(
        {
          "path": relative,
          "family": group.name,
          "input_frames": int(motion.shape[0]),
          "duration_seconds": round(duration, 6),
          "difficulty_exception": group.difficulty_exception,
          "metrics": {
            name: round(float(value), 6)
            for name, value in zip(_METRIC_NAMES, metrics, strict=True)
          },
        }
      )
    if group.target_seconds is not None and accepted_seconds < group.target_seconds:
      raise RuntimeError(
        f"{group.name} supplied only {accepted_seconds:.1f}/"
        f"{group.target_seconds:.1f} requested seconds after validation."
      )
    group_summary[group.name] = {
      "candidate_count": len(candidates),
      "selected_count": accepted,
      "rejected_during_validation": rejected,
      "target_seconds": group.target_seconds,
      "selected_seconds": round(accepted_seconds, 6),
    }

  selected_records.sort(key=lambda record: record["path"])
  excluded: list[dict[str, str]] = []
  counts: Counter[str] = Counter()
  for path in all_files:
    relative = path.relative_to(root).as_posix()
    if relative in selected_paths:
      continue
    reason = candidate_exclusions.get(relative)
    if reason is None:
      reason = (
        "family_duration_quota"
        if relative in group_for_relative
        else _outside_reason(relative)
      )
    reason_group = reason.split(":", 1)[0]
    counts[reason_group] += 1
    excluded.append({"path": relative, "reason": reason})

  total_seconds = sum(record["duration_seconds"] for record in selected_records)
  backflips = [
    record["path"]
    for record in selected_records
    if record["family"] == "back_somersault"
  ]
  return {
    "kind": "motiondecode_selection",
    "schema_version": _SCHEMA_VERSION,
    "profile": "basic-backflip-3h",
    "input_dir": str(root),
    "source_format": _source_format(),
    "selection_policy": {
      "default": "exclude",
      "deterministic_order": "sha256(relative_path)",
      "basic_kinematic_limits": {
        name: float(value)
        for name, value in zip(_METRIC_NAMES, _BASIC_LIMITS, strict=True)
      },
      "back_somersault_exception": (
        "All pure Back_Somersault takes bypass basic-motion limits but still pass "
        "schema, finite-value, quaternion, joint-position, and a dedicated "
        "[0.1, 2.5] m root-height check. Continuous forward-then-backward flips "
        "remain excluded as terrain/environment-dependent parkour."
      ),
      "note": (
        "No scene mesh is shipped with the motion CSVs. Environment/terrain/object "
        "categories are excluded because their context cannot be reconstructed from "
        "root pose plus 29 joint angles alone."
      ),
    },
    "groups": group_summary,
    "selected_count": len(selected_records),
    "selected_seconds": round(total_seconds, 6),
    "selected_hours": round(total_seconds / 3600.0, 6),
    "excluded_count": len(excluded),
    "excluded_counts_by_reason": dict(sorted(counts.items())),
    "back_somersault_count": len(backflips),
    "back_somersault_files": backflips,
    "acrobatic_exception_count": len(backflips),
    "selected_files": [record["path"] for record in selected_records],
    "selected_records": selected_records,
    "excluded_files": excluded,
  }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  try:
    with temporary.open("w") as f:
      json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-dir", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument(
    "--profile",
    choices=("basic-backflip-3h", "flat-no-props", "all-g1"),
    default="basic-backflip-3h",
  )
  parser.add_argument(
    "--exclude-prefix",
    action="append",
    default=[],
    help="Additional exact directory prefix relative to input_dir; repeatable.",
  )
  parser.add_argument(
    "--validate-headers",
    action="store_true",
    help="Validate every otherwise-selected CSV in legacy broad profiles.",
  )
  return parser


def main() -> None:
  args = _parser().parse_args()
  if args.profile == "basic-backflip-3h":
    if args.exclude_prefix:
      raise ValueError("--exclude-prefix is not supported by basic-backflip-3h.")
    payload = build_basic_backflip_selection(args.input_dir)
  else:
    payload = build_selection(
      args.input_dir,
      profile=args.profile,
      extra_prefixes=args.exclude_prefix,
      validate_headers=args.validate_headers,
    )
  output = Path(args.output)
  _write_json(output, payload)
  print(
    f"Selected {payload['selected_count']} files; excluded "
    f"{payload['excluded_count']}; wrote {output}"
  )
  if "selected_hours" in payload:
    print(
      f"  {payload['selected_hours']:.4f} h including "
      f"{payload['back_somersault_count']} pure back-somersault clips"
    )


if __name__ == "__main__":
  main()
