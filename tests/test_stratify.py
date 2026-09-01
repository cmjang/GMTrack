"""Tests for the post-Stage-I 10-second stratification manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmtrack.protocol import (
  PAPER_SHA256,
  artifact_set_sha256,
  make_stratification_protocol,
  make_stratification_provenance,
  validate_evaluation_manifest,
  validate_stage2_manifests,
)
from gmtrack.scripts.stratify import (
  _artifact_payload,
  _frame_ranges,
  _segment_entries,
  _write_json,
)


def test_sequences_at_or_below_ten_seconds_stay_whole():
  assert _frame_ranges(80, 500) == [(0, 80)]
  assert _frame_ranges(500, 500) == [(0, 500)]


def test_long_sequence_is_split_only_after_stage_one():
  assert _frame_ranges(1_230, 500) == [(0, 500), (500, 1_000), (1_000, 1_230)]


def test_short_tail_is_retained_without_overlap_or_loss():
  ranges = _frame_ranges(1_010, 500)
  assert ranges == [(0, 500), (500, 1_000), (1_000, 1_010)]
  assert sum(stop - start for start, stop in ranges) == 1_010
  assert all(stop - start <= 500 for start, stop in ranges)
  assert all(
    a_stop == b_start
    for (_, a_stop), (b_start, _) in zip(ranges, ranges[1:], strict=False)
  )


def test_single_frame_tail_borrows_from_previous_clip():
  ranges = _frame_ranges(1_001, 500)
  assert ranges == [(0, 500), (500, 999), (999, 1_001)]
  assert sum(stop - start for start, stop in ranges) == 1_001
  assert all(2 <= stop - start <= 500 for start, stop in ranges)


def test_segment_entries_reference_complete_npz_by_frame_range(tmp_path):
  source_dir = tmp_path / "source_manifests"
  output_dir = tmp_path / "derived" / "manifests"
  source_dir.mkdir(parents=True)
  output_dir.mkdir(parents=True)
  motion = tmp_path / "motions" / "walk.npz"
  motion.parent.mkdir()
  motion.touch()
  entries = [
    {
      "name": "lafan1__walk",
      "source": "lafan1",
      "path": "../motions/walk.npz",
      "num_frames": 1_010,
      "fps": 50.0,
    }
  ]

  clips = _segment_entries(entries, source_dir, output_dir, clip_seconds=10.0)

  assert [clip["name"] for clip in clips] == [
    "lafan1__walk__seg000",
    "lafan1__walk__seg001",
    "lafan1__walk__seg002",
  ]
  assert [clip["frame_start"] for clip in clips] == [0, 500, 1_000]
  assert [clip["frame_stop"] for clip in clips] == [500, 1_000, 1_010]
  assert [clip["num_frames"] for clip in clips] == [500, 500, 10]
  assert all((output_dir / clip["path"]).resolve() == motion for clip in clips)
  assert all(clip["sequence_name"] == "lafan1__walk" for clip in clips)


@pytest.mark.parametrize("num_frames,frames_per_clip", [(1, 500), (10, 1)])
def test_invalid_frame_ranges_fail_loudly(num_frames, frames_per_clip):
  with pytest.raises(ValueError):
    _frame_ranges(num_frames, frames_per_clip)


def test_segment_names_must_be_unique(tmp_path):
  entries = [
    {
      "name": "same",
      "source": "proxy",
      "path": str(tmp_path / "a.npz"),
      "num_frames": 2,
      "fps": 50,
    },
    {
      "name": "same",
      "source": "proxy",
      "path": str(tmp_path / "b.npz"),
      "num_frames": 2,
      "fps": 50,
    },
  ]
  with pytest.raises(ValueError, match="duplicate clip names"):
    _segment_entries(entries, Path("."), tmp_path, clip_seconds=10.0)


def test_nonintegral_fps_is_never_rounded_past_ten_seconds(tmp_path):
  clips = _segment_entries(
    [
      {
        "name": "proxy__walk",
        "source": "proxy",
        "path": str(tmp_path / "walk.npz"),
        "num_frames": 600,
        "fps": 29.97,
      }
    ],
    tmp_path,
    tmp_path,
    clip_seconds=10.0,
  )
  assert clips[0]["num_frames"] == 299
  assert all(clip["num_frames"] / clip["fps"] <= 10.0 for clip in clips)


def _valid_artifacts(tmp_path: Path) -> dict[str, Path]:
  source_provenance = {
    "base": "public-proxy.json",
    "note": "SEED substitutes unavailable in-house Xsens; it is not paper data.",
  }
  source = tmp_path / "input.json"
  source.write_text(
    json.dumps(
      {
        "kind": "complete_sequences",
        "provenance": source_provenance,
        "clips": [],
      }
    )
  )
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"base checkpoint")
  protocol = make_stratification_protocol(seed=17)
  provenance = make_stratification_provenance(
    source,
    checkpoint,
    dataset_label="v1-proxy",
    input_manifest_provenance=source_provenance,
  )
  paths = {
    "stratified": tmp_path / "stratified.json",
    "mastered": tmp_path / "mastered.json",
    "challenging": tmp_path / "challenging.json",
    "report": tmp_path / "stratification_report.json",
  }
  clips = [
    {
      "name": "proxy__easy",
      "source": "seed-proxy",
      "path": "easy.npz",
      "sequence_name": "proxy__easy",
      "frame_start": 0,
      "frame_stop": 500,
      "num_frames": 500,
      "fps": 50.0,
      "trials": 5,
      "success_count": 4,
      "success_rate": 0.8,
      "classification": "mastered",
    },
    {
      "name": "proxy__hard",
      "source": "seed-proxy",
      "path": "hard.npz",
      "sequence_name": "proxy__hard",
      "frame_start": 20,
      "frame_stop": 270,
      "num_frames": 250,
      "fps": 50.0,
      "trials": 5,
      "success_count": 3,
      "success_rate": 0.6,
      "classification": "challenging",
    },
  ]

  def record(clip: dict, successes: int, classification: str) -> dict:
    rate = successes / 5
    return {
      "trials": 5,
      "successes": successes,
      "success_count": successes,
      "success_rate": rate,
      "completion_rate": rate,
      "classification": classification,
      "source": clip["source"],
      "sequence_name": clip["sequence_name"],
      "frame_start": clip["frame_start"],
      "frame_stop": clip["frame_stop"],
      "num_frames": clip["num_frames"],
      "fps": clip["fps"],
      "duration_seconds": clip["num_frames"] / clip["fps"],
    }

  payload_specs = {
    "stratified": ("stratification_clips", clips),
    "mastered": ("mastered_clips", [clips[0]]),
    "challenging": ("challenging_clips", [clips[1]]),
    "report": (
      "stratification_report",
      {
        clips[0]["name"]: record(clips[0], 4, "mastered"),
        clips[1]["name"]: record(clips[1], 3, "challenging"),
      },
    ),
  }
  artifact_id = artifact_set_sha256(
    protocol,
    provenance,
    {label: artifact_clips for label, (_, artifact_clips) in payload_specs.items()},
  )
  for label, (kind, artifact_clips) in payload_specs.items():
    _write_json(
      paths[label],
      _artifact_payload(
        kind=kind,
        owner=paths[label],
        artifact_paths=paths,
        protocol=protocol,
        provenance=provenance,
        artifact_id=artifact_id,
        clips=artifact_clips,
      ),
    )
  return paths


def _mutate_json(path: Path, mutation) -> None:
  payload = json.loads(path.read_text())
  mutation(payload)
  path.write_text(json.dumps(payload))


def test_strict_v1_artifacts_cross_validate_and_return_report(tmp_path):
  paths = _valid_artifacts(tmp_path)

  report = validate_stage2_manifests(
    paths["stratified"], paths["mastered"], paths["challenging"]
  )

  assert report["provenance"]["paper_sha256"] == PAPER_SHA256
  assert report["provenance"]["dataset_label"] == "v1-proxy"
  assert report["provenance"]["input_manifest"]["declared_provenance"]
  assert report["clips"]["proxy__easy"]["success_count"] == 4


@pytest.mark.parametrize("label", ["stratified", "mastered", "challenging"])
def test_evaluation_manifest_must_belong_to_validated_artifact_set(tmp_path, label):
  paths = _valid_artifacts(tmp_path)

  validation = validate_evaluation_manifest(paths[label])

  assert validation["validated"] is True
  assert validation["manifest_kind"] == label
  assert validation["paper_sha256"] == PAPER_SHA256


@pytest.mark.parametrize(
  ("field", "value", "message"),
  [
    ("trials", 4, "exactly 5 trials"),
    ("threshold", 0.6, "threshold of 0.8"),
    ("max_clip_seconds", 11.0, "maximum clip duration of 10"),
  ],
)
def test_validator_rejects_non_v1_protocol(tmp_path, field, value, message):
  paths = _valid_artifacts(tmp_path)
  _mutate_json(
    paths["stratified"], lambda payload: payload["protocol"].__setitem__(field, value)
  )

  with pytest.raises(ValueError, match=message):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_more_than_ten_seconds(tmp_path):
  paths = _valid_artifacts(tmp_path)

  def make_long(payload):
    payload["clips"][0]["frame_stop"] = 501
    payload["clips"][0]["num_frames"] = 501

  _mutate_json(paths["stratified"], make_long)
  with pytest.raises(ValueError, match="10-second maximum"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_duplicate_names(tmp_path):
  paths = _valid_artifacts(tmp_path)
  _mutate_json(
    paths["stratified"], lambda payload: payload["clips"].append(payload["clips"][0])
  )
  with pytest.raises(ValueError, match="duplicate clip name"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_dm_dc_overlap(tmp_path):
  paths = _valid_artifacts(tmp_path)
  mastered_clip = json.loads(paths["mastered"].read_text())["clips"][0]
  _mutate_json(
    paths["challenging"], lambda payload: payload["clips"].append(mastered_clip)
  )
  with pytest.raises(ValueError, match="D_m and D_c overlap"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_incomplete_dm_dc_cover(tmp_path):
  paths = _valid_artifacts(tmp_path)
  _mutate_json(paths["challenging"], lambda payload: payload.__setitem__("clips", []))
  with pytest.raises(ValueError, match="disjoint cover"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_coherent_clip_content_tampering(tmp_path):
  paths = _valid_artifacts(tmp_path)

  def change_source(payload):
    clips = payload["clips"]
    if isinstance(clips, list):
      for clip in clips:
        if clip["name"] == "proxy__easy":
          clip["source"] = "tampered-proxy"
    else:
      clips["proxy__easy"]["source"] = "tampered-proxy"

  for label in ("stratified", "mastered", "report"):
    _mutate_json(paths[label], change_source)

  with pytest.raises(ValueError, match="exact stratified/D_m/D_c/report clip content"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_missing_hash_provenance(tmp_path):
  paths = _valid_artifacts(tmp_path)
  _mutate_json(
    paths["stratified"],
    lambda payload: payload["provenance"].pop("base_checkpoint_sha256"),
  )
  with pytest.raises(ValueError, match="base_checkpoint_sha256"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_success_rate_or_frame_bound_mismatch(tmp_path):
  paths = _valid_artifacts(tmp_path)

  def corrupt_report(payload):
    payload["clips"]["proxy__easy"]["success_rate"] = 0.6
    payload["clips"]["proxy__easy"]["frame_stop"] = 499

  _mutate_json(paths["report"], corrupt_report)
  with pytest.raises(ValueError, match="success_rate disagrees"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_rejects_exact_paper_dataset_claim(tmp_path):
  paths = _valid_artifacts(tmp_path)
  _mutate_json(
    paths["stratified"],
    lambda payload: payload["provenance"].__setitem__(
      "dataset_label", "exact-paper-data"
    ),
  )
  with pytest.raises(ValueError, match="v1-proxy"):
    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )


def test_validator_cannot_override_the_unique_v1_paper_hash(tmp_path):
  paths = _valid_artifacts(tmp_path)
  with pytest.raises(ValueError, match="must remain pinned"):
    validate_stage2_manifests(
      paths["stratified"],
      paths["mastered"],
      paths["challenging"],
      expected_paper_sha256="0" * 64,
    )
