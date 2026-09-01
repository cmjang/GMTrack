"""Tests for MotionDecode taxonomy and narrow training curation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gmtrack.scripts.curate_motiondecode as curate
from gmtrack.scripts.curate_motiondecode import (
  BasicGroup,
  _backflip_candidate,
  _kinematic_metrics,
  _matches_prefix,
  build_basic_backflip_selection,
  build_selection,
)
from gmtrack.scripts.prepare_motions import MOTIONDECODE_HEADER


def _csv(root: Path, relative: str, *, valid: bool = True) -> Path:
  path = root / relative
  path.parent.mkdir(parents=True, exist_ok=True)
  if not valid:
    path.write_text("not,the,g1,schema\n")
    return path
  motion = np.zeros((6, 36), dtype=np.float64)
  motion[:, 2] = 0.8
  motion[:, 3] = 1.0
  with path.open("w") as f:
    f.write(",".join(MOTIONDECODE_HEADER) + "\n")
    np.savetxt(f, motion, delimiter=",")
  return path


def test_flat_profile_excludes_environment_but_keeps_back_somersault(tmp_path):
  normal = "samples/1.3/1.3.1/Walk/normal.csv"
  backflip = (
    "samples/2.3.Extreme_Environment_Interaction/2.3.1.Parkour_Actions/"
    "2.3.1.12.Back_Somersault/back.csv"
  )
  wall_flip = (
    "samples/2.3.Extreme_Environment_Interaction/2.3.1.Parkour_Actions/"
    "2.3.1.14.Wall_Flip/wall.csv"
  )
  stairs = (
    "samples/2.1.Basic_Environment_Interaction/"
    "2.1.1.Stair_and_Step_Climbing_Descending/stairs.csv"
  )
  for relative in (normal, backflip, wall_flip, stairs):
    _csv(tmp_path, relative)

  manifest = build_selection(tmp_path, validate_headers=True)
  assert manifest["selected_files"] == [normal, backflip]
  assert manifest["backflip_candidates"] == [backflip]
  assert manifest["excluded_count"] == 2
  by_path = {entry["path"]: entry["reason"] for entry in manifest["excluded_files"]}
  assert by_path[wall_flip] == "parkour_prop_or_height"
  assert by_path[stairs] == "stairs_steps"


def test_all_g1_profile_only_applies_explicit_extra_prefixes(tmp_path):
  keep = "samples/a/keep.csv"
  drop = "samples/drop/drop.csv"
  _csv(tmp_path, keep)
  _csv(tmp_path, drop)
  manifest = build_selection(
    tmp_path, profile="all-g1", extra_prefixes=("samples/drop",)
  )
  assert manifest["selected_files"] == [keep]
  assert manifest["excluded_files"] == [{"path": drop, "reason": "custom"}]


def test_prefix_matching_has_a_path_boundary():
  assert _matches_prefix("samples/stairs/a.csv", "samples/stairs")
  assert not _matches_prefix("samples/stairs_extra/a.csv", "samples/stairs")


def test_backflip_aliases_do_not_match_unrelated_wall_flip():
  assert _backflip_candidate("samples/Back_Somersault/a.csv")
  assert _backflip_candidate("samples/Backward-Flip/a.csv")
  assert not _backflip_candidate("samples/Wall_Flip/a.csv")


def test_acrobatic_root_height_bound_is_separate_from_basic_bound():
  motion = np.zeros((6, 36), dtype=np.float64)
  motion[:, 2] = 2.35
  motion[:, 3] = 1.0
  with np.testing.assert_raises_regex(ValueError, r"\[0.1, 2.0\]"):
    _kinematic_metrics(motion)
  assert _kinematic_metrics(motion, root_height_max=2.5).shape == (5,)


def test_optional_header_validation_excludes_non_g1_csv(tmp_path):
  valid = "samples/valid/motion.csv"
  invalid = "samples/invalid/motion.csv"
  _csv(tmp_path, valid)
  _csv(tmp_path, invalid, valid=False)
  manifest = build_selection(tmp_path, profile="all-g1", validate_headers=True)
  assert manifest["selected_files"] == [valid]
  assert manifest["excluded_files"] == [
    {"path": invalid, "reason": "incompatible_schema"}
  ]
  assert manifest["header_validation"] == "all_selected"


def test_narrow_profile_keeps_pure_backflip_and_denies_continuous_flip(
  tmp_path, monkeypatch
):
  walk_prefix = "samples/easy/Walk"
  back_prefix = "samples/parkour/Back_Somersault"
  walk = f"{walk_prefix}/walk.csv"
  back = f"{back_prefix}/back.csv"
  compound = "samples/parkour/Forward_Somersault_with_Backward_Flip/compound.csv"
  outside = "samples/object/carry.csv"
  for relative in (walk, back, compound, outside):
    _csv(tmp_path, relative)
  monkeypatch.setattr(
    curate,
    "BASIC_GROUPS",
    (
      BasicGroup("walk", (walk_prefix,), 0.01),
      BasicGroup("back_somersault", (back_prefix,), None, True),
    ),
  )

  manifest = build_basic_backflip_selection(tmp_path)
  assert manifest["selected_files"] == [walk, back]
  assert manifest["back_somersault_files"] == [back]
  assert manifest["acrobatic_exception_count"] == 1
  assert manifest["selected_count"] == 2
  assert manifest["excluded_files"] == [
    {"path": outside, "reason": "outside_basic_allowlist"},
    {"path": compound, "reason": "outside_basic_allowlist"},
  ]
