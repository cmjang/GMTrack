"""Curate an AMASS-sized easy/moderate subset from the local SEED archive.

``data/processed`` is a legacy BONES-SEED export: root/body positions are in
centimetres, joint positions and angular velocities are in degrees, and body
quaternions are already wxyz.  It is not a training-ready motion library.  This
script selects balanced original/mirrored pairs, converts the root pose and joints
to the SI/xyzw CSV convention consumed by :mod:`prepare_motions`, and records an
auditable selection report.

The generated CSVs must still pass through ``prepare_motions`` so MuJoCo recomputes
body ordering and all velocities using the current G1 model::

  python -m gmtrack.scripts.curate_seed_simple curate
  python -m gmtrack.scripts.prepare_motions \
    --input-dir data/datasets/raw/seed_simple --source seed-simple --input-fps 50 \
    --output-fps 50 --input-format mjlab \
    --output-dir data/datasets/stage1_full/seed_simple \
    --manifest logs/data_build/manifests/seed_simple_full.json
  python -m gmtrack.scripts.curate_seed_simple merge
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Family = Literal["steady_walk", "walk_transition", "jog", "gesture"]

_PAIR_RE = re.compile(
  r"^seed__(?P<action>.+)__A(?P<actor>\d+)"
  r"(?P<mirror>_M)?(?P<chunk>__\d+)?\.npz$"
)

# 92 mirrored pairs -> 184 source clips.  The current converter turns each
# 500-frame source into 499 output frames, i.e. 0.5101 h at 50 Hz, within 0.2% of
# the paper's 0.511 h AMASS allocation.
_PAIR_QUOTAS: dict[Family, int] = {
  "steady_walk": 20,
  "walk_transition": 26,
  "jog": 18,
  "gesture": 28,
}

_TARGET_DIFFICULTY: dict[Family, float] = {
  "steady_walk": 0.30,
  "walk_transition": 0.25,
  "jog": 0.40,
  "gesture": 0.22,
}

# P99 limits in SI units after correcting the legacy export.  They reject flips,
# discontinuities, and violent contacts while retaining normal walk/jog/turn and
# upper-body motion.
_LIMITS = np.array([8.0, 200.0, 6.0, 5.0, 45.0], dtype=np.float64)
_METRIC_NAMES = (
  "joint_speed_p99_rad_s",
  "joint_acc_p99_rad_s2",
  "root_ang_speed_p99_rad_s",
  "body_speed_p99_m_s",
  "root_tilt_p99_deg",
)


@dataclass(frozen=True)
class PairCandidate:
  family: Family
  action: str
  actor: str
  chunk: str
  original: Path
  mirrored: Path
  metrics: np.ndarray
  root_xy_speed_p50: float
  difficulty: float


def _family(action: str) -> Family | None:
  name = action.lower()
  excluded = (
    "object",
    "horse",
    "sit_",
    "jump",
    "flip",
    "roll",
    "cartwheel",
    "kick",
    "throw",
    "dance",
    "fight",
    "attack",
    "combat",
    "bump",
    "avoid",
    "dust",
    "body_check",
    "body_search",
  )
  if any(token in name for token in excluded):
    return None
  if "jog" in name:
    return "jog"
  if any(
    token in name
    for token in (
      "neutral_walk",
      "relaxed_walk",
      "loop_forward_walk",
      "loop_backward_walk",
      "sideway_walk",
      "arc_walk_left_loop",
    )
  ):
    return "steady_walk"
  if any(
    token in name
    for token in ("walk", "idle", "stance_change", "step_rotate", "stoop_down")
  ):
    return "walk_transition"
  if any(
    token in name
    for token in (
      "body_stretch",
      "reaching_",
      "looking_",
      "bow_",
      "salute_",
      "yawn_",
      "thinking_",
      "rubbing_hands",
      "show_bicep",
      "welcoming_",
      "triumph_",
      "sneeze_",
      "freezing_cold",
      "beckon_",
      "alone_",
      "lamenting_",
    )
  ):
    return "gesture"
  return None


def _legacy_metrics(path: Path) -> tuple[np.ndarray, float] | None:
  with np.load(path) as data:
    required = {
      "fps",
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    }
    if not required.issubset(data.files):
      return None
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if not math.isclose(fps, 50.0) or data["joint_pos"].shape != (500, 29):
      return None

    joint_pos = np.deg2rad(data["joint_pos"])
    joint_vel = np.deg2rad(data["joint_vel"])
    body_pos = data["body_pos_w"] * 0.01
    body_lin_vel = data["body_lin_vel_w"] * 0.01
    body_ang_vel = np.deg2rad(data["body_ang_vel_w"])
    root_quat = data["body_quat_w"][:, 0]

  arrays = (joint_pos, joint_vel, body_pos, body_lin_vel, body_ang_vel, root_quat)
  if not all(np.isfinite(array).all() for array in arrays):
    return None
  quat_norm = np.linalg.norm(root_quat, axis=1)
  if not np.allclose(quat_norm, 1.0, atol=1.0e-3):
    return None
  if np.max(np.abs(joint_pos)) > 3.2:
    return None
  root_height = body_pos[:, 0, 2]
  if root_height.min() < 0.35 or root_height.max() > 1.30:
    return None

  joint_acc = np.abs(np.diff(joint_vel, axis=0)) * fps
  root_quat_x = root_quat[:, 1]
  root_quat_y = root_quat[:, 2]
  root_z_alignment = np.clip(
    1.0 - 2.0 * (np.square(root_quat_x) + np.square(root_quat_y)), -1.0, 1.0
  )
  root_tilt = np.rad2deg(np.arccos(root_z_alignment))
  root_xy_speed = np.linalg.norm(body_lin_vel[:, 0, :2], axis=1)

  metrics = np.array(
    [
      np.quantile(np.abs(joint_vel), 0.99),
      np.quantile(joint_acc, 0.99),
      np.quantile(np.linalg.norm(body_ang_vel[:, 0], axis=1), 0.99),
      np.quantile(np.linalg.norm(body_lin_vel, axis=2), 0.99),
      np.quantile(root_tilt, 0.99),
    ],
    dtype=np.float64,
  )
  return metrics, float(np.quantile(root_xy_speed, 0.5))


def _candidates(input_dir: Path) -> list[PairCandidate]:
  grouped: dict[tuple[Family, str, str, str], dict[str, Path]] = defaultdict(dict)
  for path in sorted(input_dir.glob("seed__*.npz")):
    match = _PAIR_RE.match(path.name)
    if match is None:
      continue
    family = _family(match["action"])
    if family is None:
      continue
    key = (family, match["action"], match["actor"], match["chunk"] or "")
    grouped[key]["mirrored" if match["mirror"] else "original"] = path

  candidates: list[PairCandidate] = []
  for (family, action, actor, chunk), pair in grouped.items():
    if pair.keys() != {"original", "mirrored"}:
      continue
    original_metrics = _legacy_metrics(pair["original"])
    mirrored_metrics = _legacy_metrics(pair["mirrored"])
    if original_metrics is None or mirrored_metrics is None:
      continue
    metrics = np.maximum(original_metrics[0], mirrored_metrics[0])
    root_xy_speed_p50 = max(original_metrics[1], mirrored_metrics[1])
    if np.any(metrics > _LIMITS):
      continue
    if family == "steady_walk" and root_xy_speed_p50 < 0.20:
      continue
    if family == "jog" and root_xy_speed_p50 < 0.50:
      continue
    candidates.append(
      PairCandidate(
        family=family,
        action=action,
        actor=actor,
        chunk=chunk,
        original=pair["original"],
        mirrored=pair["mirrored"],
        metrics=metrics,
        root_xy_speed_p50=root_xy_speed_p50,
        difficulty=float(np.mean(metrics / _LIMITS)),
      )
    )
  return candidates


def _select(candidates: list[PairCandidate]) -> list[PairCandidate]:
  selected: list[PairCandidate] = []
  for family, quota in _PAIR_QUOTAS.items():
    by_action: dict[str, list[PairCandidate]] = defaultdict(list)
    for candidate in candidates:
      if candidate.family == family:
        by_action[candidate.action].append(candidate)

    primary: list[PairCandidate] = []
    extras: list[PairCandidate] = []
    target = _TARGET_DIFFICULTY[family]
    for action_candidates in by_action.values():
      action_candidates.sort(
        key=lambda item: (
          abs(item.difficulty - target),
          item.difficulty,
          item.actor,
          item.chunk,
        )
      )
      primary.append(action_candidates[0])
      extras.extend(action_candidates[1:])

    primary.sort(
      key=lambda item: (item.difficulty, item.action, item.actor, item.chunk)
    )
    family_selection = primary[:quota]
    if len(family_selection) < quota:
      action_counts = Counter(item.action for item in family_selection)
      extras.sort(
        key=lambda item: (
          action_counts[item.action],
          abs(item.difficulty - target),
          item.action,
          item.actor,
          item.chunk,
        )
      )
      for candidate in extras:
        if len(family_selection) >= quota:
          break
        if action_counts[candidate.action] >= 2:
          continue
        family_selection.append(candidate)
        action_counts[candidate.action] += 1

    if len(family_selection) != quota:
      raise RuntimeError(
        f"Could select only {len(family_selection)}/{quota} mirrored pairs for "
        f"{family}."
      )
    selected.extend(family_selection)
  return sorted(
    selected, key=lambda item: (item.family, item.action, item.actor, item.chunk)
  )


def _training_csv(path: Path) -> np.ndarray:
  with np.load(path) as data:
    root_pos = data["body_pos_w"][:, 0] * 0.01
    root_quat_wxyz = data["body_quat_w"][:, 0]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    joint_pos = np.deg2rad(data["joint_pos"])
  return np.concatenate((root_pos, root_quat_xyzw, joint_pos), axis=1).astype(
    np.float32
  )


def _report_entry(
  candidate: PairCandidate, path: Path, output_dir: Path
) -> dict[str, Any]:
  return {
    "name": path.stem.removeprefix("seed__"),
    "family": candidate.family,
    "action": candidate.action,
    "actor": candidate.actor,
    "mirrored": "_M" in path.stem.rsplit("__A", maxsplit=1)[-1],
    "legacy_path": os.path.relpath(path.resolve(), Path.cwd()),
    "csv_path": os.path.relpath(
      (output_dir / f"{path.stem.removeprefix('seed__')}.csv").resolve(), Path.cwd()
    ),
    "num_source_frames": 500,
    "fps": 50.0,
    "difficulty": candidate.difficulty,
    "root_xy_speed_p50_m_s": candidate.root_xy_speed_p50,
    "metrics": {
      name: float(value)
      for name, value in zip(_METRIC_NAMES, candidate.metrics, strict=True)
    },
  }


def curate(args: argparse.Namespace) -> None:
  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  report_path = Path(args.report)
  if not input_dir.is_dir():
    raise FileNotFoundError(input_dir)

  selected = _select(_candidates(input_dir))
  selected_paths = {
    path for candidate in selected for path in (candidate.original, candidate.mirrored)
  }
  expected_names = {
    f"{path.stem.removeprefix('seed__')}.csv" for path in selected_paths
  }
  output_dir.mkdir(parents=True, exist_ok=True)
  unexpected = {path.name for path in output_dir.glob("*.csv")} - expected_names
  if unexpected:
    examples = ", ".join(sorted(unexpected)[:5])
    raise RuntimeError(
      f"{output_dir} contains {len(unexpected)} unrelated CSV(s): {examples}"
    )

  report_entries: list[dict[str, Any]] = []
  by_path = {
    path: candidate
    for candidate in selected
    for path in (candidate.original, candidate.mirrored)
  }
  for path in sorted(selected_paths):
    destination = output_dir / f"{path.stem.removeprefix('seed__')}.csv"
    if args.overwrite or not destination.exists():
      np.savetxt(destination, _training_csv(path), delimiter=",", fmt="%.9g")
    report_entries.append(_report_entry(by_path[path], path, output_dir))

  report_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "schema_version": 1,
    "purpose": "Easy/moderate SEED substitute for the paper's 0.511 h AMASS slot",
    "source_format": {
      "positions": "centimetres",
      "joint_and_angular_values": "degrees",
      "body_quaternion": "wxyz",
    },
    "selection": {
      "pair_quotas": _PAIR_QUOTAS,
      "metric_limits": dict(zip(_METRIC_NAMES, _LIMITS.tolist(), strict=True)),
      "source_clips": len(report_entries),
      "source_hours": len(report_entries) * 500 / 50.0 / 3600.0,
      "expected_converted_hours": len(report_entries) * 499 / 50.0 / 3600.0,
    },
    "clips": report_entries,
  }
  temporary = report_path.with_suffix(f"{report_path.suffix}.tmp")
  with temporary.open("w") as file:
    json.dump(payload, file, indent=2)
  os.replace(temporary, report_path)

  counts = Counter(candidate.family for candidate in selected)
  print(
    f"[gmtrack] selected {len(selected)} mirrored pairs / {len(report_entries)} "
    f"clips; expected converted duration "
    f"{payload['selection']['expected_converted_hours']:.4f} h"
  )
  print(f"[gmtrack] pair quotas: {dict(counts)}")
  print(f"[gmtrack] CSVs: {output_dir}")
  print(f"[gmtrack] report: {report_path}")


def merge(args: argparse.Namespace) -> None:
  base_path = Path(args.base_manifest)
  additional_path = Path(args.additional_manifest)
  output_path = Path(args.output_manifest)
  with base_path.open() as file:
    base_payload = json.load(file)
  with additional_path.open() as file:
    additional_payload = json.load(file)
  for label, path, payload in (
    ("base", base_path, base_payload),
    ("additional", additional_path, additional_payload),
  ):
    if payload.get("kind") != "complete_sequences":
      raise ValueError(
        f"The {label} manifest {path} is not a complete-sequence manifest. "
        "Regenerate it with prepare_motions."
      )
  base = base_payload["clips"]
  additional = additional_payload["clips"]

  clips = [clip for clip in base if clip["source"] == args.base_source]
  clips.extend(additional)
  by_name = {clip["name"]: clip for clip in clips}
  if len(by_name) != len(clips):
    raise ValueError("The merged manifests contain duplicate clip names.")
  clips = sorted(clips, key=lambda clip: clip["name"])

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
  with temporary.open("w") as file:
    json.dump({"kind": "complete_sequences", "clips": clips}, file, indent=2)
  os.replace(temporary, output_path)
  duration = sum(clip["num_frames"] / clip["fps"] for clip in clips) / 3600.0
  sources = Counter(clip["source"] for clip in clips)
  print(
    f"[gmtrack] merged {len(clips)} clips / {duration:.4f} h to {output_path}; "
    f"sources={dict(sources)}"
  )


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)

  curate_parser = subparsers.add_parser("curate")
  curate_parser.add_argument("--input-dir", default="data/processed")
  curate_parser.add_argument("--output-dir", default="data/datasets/raw/seed_simple")
  curate_parser.add_argument(
    "--report", default="logs/data_qc/seed_simple_selection.json"
  )
  curate_parser.add_argument("--overwrite", action="store_true")
  curate_parser.set_defaults(func=curate)

  merge_parser = subparsers.add_parser("merge")
  merge_parser.add_argument(
    "--base-manifest", default="logs/data_build/manifests/lafan1_full.json"
  )
  merge_parser.add_argument("--base-source", default="lafan1")
  merge_parser.add_argument(
    "--additional-manifest", default="logs/data_build/manifests/seed_simple_full.json"
  )
  merge_parser.add_argument(
    "--output-manifest",
    default="logs/data_build/manifests/stage1_lafan_seed_simple.json",
  )
  merge_parser.set_defaults(func=merge)
  return parser


def main() -> None:
  args = _parser().parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
