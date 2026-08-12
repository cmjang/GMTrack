"""Tests for diversity-preserving SEED cartwheel curation."""

import json

import pytest

from ex_grmt.scripts.curate_seed_cartwheel import (
  build_balanced_manifest,
  build_heldout_manifest,
  select_balanced_pairs,
)


def _clip(family: str, actor: int, frames: int, *, mirror: bool = False) -> dict:
  suffix = "_M" if mirror else ""
  name = f"seed-stunts__{family}__A{actor}{suffix}"
  return {
    "name": name,
    "source": "seed-stunts",
    "path": f"../motions/{name}.npz",
    "num_frames": frames,
    "fps": 50,
  }


def _pair(family: str, actor: int, frames: int) -> list[dict]:
  return [
    _clip(family, actor, frames),
    _clip(family, actor, frames, mirror=True),
  ]


def test_selects_every_r_family_then_adds_from_the_most_populated():
  clips = [
    *_pair("cartwheel_R_001", 1, 100),
    *_pair("cartwheel_R_001", 2, 200),
    *_pair("cartwheel_R_001", 3, 400),
    *_pair("cartwheel_R_002", 8, 300),
    *_pair("cartwheelin_001", 9, 350),
  ]

  selected, source_counts, selected_counts = select_balanced_pairs(
    clips, target_unique_takes=3
  )

  assert [clip["name"] for clip in selected] == [
    "seed-stunts__cartwheel_R_001__A1",
    "seed-stunts__cartwheel_R_001__A1_M",
    "seed-stunts__cartwheel_R_001__A2",
    "seed-stunts__cartwheel_R_001__A2_M",
    "seed-stunts__cartwheel_R_002__A8",
    "seed-stunts__cartwheel_R_002__A8_M",
  ]
  assert source_counts == {"cartwheel_R_001": 3, "cartwheel_R_002": 1}
  assert selected_counts == {"cartwheel_R_001": 2, "cartwheel_R_002": 1}


def test_rejects_an_incomplete_mirror_pair():
  with pytest.raises(ValueError, match="Incomplete original/mirror pair"):
    select_balanced_pairs([_clip("cartwheel_R_001", 1, 100)], target_unique_takes=1)


def test_distributes_extra_slots_across_families_before_third_take():
  clips = [
    *_pair("cartwheel_R_001", 1, 100),
    *_pair("cartwheel_R_001", 2, 200),
    *_pair("cartwheel_R_001", 3, 300),
    *_pair("cartwheel_R_002", 4, 100),
    *_pair("cartwheel_R_002", 5, 200),
  ]

  _, _, selected_counts = select_balanced_pairs(clips, target_unique_takes=4)

  assert selected_counts == {"cartwheel_R_001": 2, "cartwheel_R_002": 2}


def test_build_records_selection_and_rebases_paths(tmp_path):
  source_dir = tmp_path / "source"
  output_dir = tmp_path / "curated"
  motions = tmp_path / "motions"
  source_dir.mkdir()
  motions.mkdir()
  clips = _pair("cartwheel_R_001", 1, 100)
  for clip in clips:
    (motions / f"{clip['name']}.npz").touch()
  source = source_dir / "all.json"
  source.write_text(json.dumps({"kind": "complete_sequences", "clips": clips}))

  result = build_balanced_manifest(
    source, output_dir / "balanced.json", target_unique_takes=1
  )

  assert result["clip_count"] == 2
  assert result["total_seconds"] == 4.0
  assert result["provenance"]["unique_after_unmirroring"] == 1
  assert result["clips"][0]["path"] == "../motions/seed-stunts__cartwheel_R_001__A1.npz"


def test_build_heldout_is_disjoint_and_exhaustive(tmp_path):
  source_dir = tmp_path / "source"
  selected_dir = tmp_path / "selected"
  output_dir = tmp_path / "test"
  motions = tmp_path / "motions"
  source_dir.mkdir()
  selected_dir.mkdir()
  motions.mkdir()
  source_clips = [
    *_pair("cartwheel_R_001", 1, 100),
    *_pair("cartwheel_R_002", 2, 200),
  ]
  for clip in source_clips:
    (motions / f"{clip['name']}.npz").touch()
  source = source_dir / "all.json"
  selected = selected_dir / "training.json"
  source.write_text(json.dumps({"kind": "complete_sequences", "clips": source_clips}))
  selected.write_text(
    json.dumps({"kind": "complete_sequences", "clips": source_clips[:2]})
  )

  result = build_heldout_manifest(source, selected, output_dir / "heldout.json")

  assert result["clip_count"] == 2
  assert result["total_seconds"] == 8.0
  assert result["provenance"]["source_clip_count"] == 4
  assert result["provenance"]["excluded_training_clip_count"] == 2
  assert result["provenance"]["name_overlap_with_training"] == 0
  assert {clip["name"] for clip in result["clips"]}.isdisjoint(
    {clip["name"] for clip in source_clips[:2]}
  )
  assert result["clips"][0]["path"] == "../motions/seed-stunts__cartwheel_R_002__A2.npz"
