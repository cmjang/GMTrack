"""Strict, dependency-free provenance checks for Stage-II stratification.

The Stage-II task imports this module while constructing its motion command, so it
must remain free of mjlab, MuJoCo, and torch imports.  A malformed or stale set of
manifests is rejected before any motion data is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

STRATIFICATION_SCHEMA = "gmtrack.stage2_stratification"
STRATIFICATION_SCHEMA_VERSION = 1
STRATIFICATION_PROTOCOL = "gmtrack-stage2-stratification-v1"

PAPER_ARXIV_ID = "2607.20110v1"
PAPER_SHA256 = "55cca5c02f16c659e4ab3baf08d9ad1fb69865f37cfba084958ade0911cf51fe"

# Stage-II stratification defaults.
STRATIFICATION_TRIALS = 5
STRATIFICATION_THRESHOLD = 0.8
STRATIFICATION_MAX_CLIP_SECONDS = 10.0

_RANDOMNESS_SOURCES = (
  "domain_randomization",
  "external_pushes",
  "observation_noise",
)
_ARTIFACT_KINDS = {
  "stratified": "stratification_clips",
  "mastered": "mastered_clips",
  "challenging": "challenging_clips",
  "report": "stratification_report",
}


def sha256_file(path: str | Path) -> str:
  """Return the SHA256 of a file's exact bytes."""
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def make_stratification_protocol(seed: int) -> dict[str, Any]:
  """Build the immutable v1 protocol block recorded in every artifact."""
  if isinstance(seed, bool) or not isinstance(seed, int):
    raise TypeError(f"Stratification seed must be an integer, got {seed!r}.")
  if not 0 <= seed <= 2**31 - 1:
    raise ValueError(f"Stratification seed must be in [0, 2^31 - 1], got {seed}.")
  return {
    "id": STRATIFICATION_PROTOCOL,
    "version": STRATIFICATION_SCHEMA_VERSION,
    "trials": STRATIFICATION_TRIALS,
    "threshold": STRATIFICATION_THRESHOLD,
    "max_clip_seconds": STRATIFICATION_MAX_CLIP_SECONDS,
    "success_criterion": {
      "id": "root_height_deviation",
      "tolerance_m": 0.2,
    },
    "randomness": {
      "seed": seed,
      "randomized_rollouts": True,
      "sources": list(_RANDOMNESS_SOURCES),
      "start_frame": 0,
      "policy_actions": "deterministic",
      "determinism": "best_effort_mujoco_warp",
    },
  }


def make_stratification_provenance(
  input_manifest: str | Path,
  base_checkpoint: str | Path,
  *,
  dataset_label: str,
  input_manifest_provenance: dict[str, Any],
) -> dict[str, Any]:
  """Hash the two run inputs and identify the sole authoritative paper artifact."""
  if dataset_label != "v1-proxy":
    raise ValueError(
      "Strict v1 artifacts must use dataset_label='v1-proxy'; the unavailable "
      "AMASS/in-house Xsens corpus must not be claimed as exact paper data."
    )
  if not isinstance(input_manifest_provenance, dict) or not input_manifest_provenance:
    raise ValueError("The input manifest must contain non-empty proxy provenance.")
  input_path = Path(input_manifest)
  checkpoint_path = Path(base_checkpoint)
  input_sha256 = sha256_file(input_path)
  checkpoint_sha256 = sha256_file(checkpoint_path)
  return {
    "paper": {
      "arxiv_id": PAPER_ARXIV_ID,
      "sha256": PAPER_SHA256,
    },
    "paper_sha256": PAPER_SHA256,
    "input_manifest": {
      "path": str(input_path),
      "sha256": input_sha256,
      "declared_provenance": input_manifest_provenance,
    },
    "input_manifest_sha256": input_sha256,
    "base_checkpoint": {
      "path": str(checkpoint_path),
      "sha256": checkpoint_sha256,
    },
    "base_checkpoint_sha256": checkpoint_sha256,
    "dataset_label": dataset_label.strip(),
  }


def artifact_set_sha256(
  protocol: dict[str, Any],
  provenance: dict[str, Any],
  artifact_clips: dict[str, Any],
) -> str:
  """Content identifier shared by all four artifacts from one stratification run.

  The digest covers the actual clip payload of every artifact, not just common
  metadata.  A post-hoc edit to D_m, D_c, the full set, or the rollout report is
  therefore detected even when the edited files remain mutually consistent.
  """
  if set(artifact_clips) != set(_ARTIFACT_KINDS):
    missing = set(_ARTIFACT_KINDS) - set(artifact_clips)
    extra = set(artifact_clips) - set(_ARTIFACT_KINDS)
    raise ValueError(
      "artifact_clips must contain exactly stratified/mastered/challenging/report; "
      f"missing={sorted(missing)!r}, extra={sorted(extra)!r}."
    )
  canonical = json.dumps(
    {
      "protocol": protocol,
      "provenance": provenance,
      "artifact_clips": {
        label: artifact_clips[label] for label in sorted(_ARTIFACT_KINDS)
      },
    },
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
  ).encode()
  return hashlib.sha256(canonical).hexdigest()


def _fail(message: str) -> None:
  raise ValueError(f"Invalid Stage-II stratification artifacts: {message}")


def _mapping(value: object, where: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    _fail(f"{where} must be a JSON object.")
  return value


def _integer(value: object, where: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    _fail(f"{where} must be an integer, got {value!r}.")
  return value


def _number(value: object, where: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    _fail(f"{where} must be a number, got {value!r}.")
  result = float(value)
  if not math.isfinite(result):
    _fail(f"{where} must be finite, got {value!r}.")
  return result


def _sha256(value: object, where: str) -> str:
  if (
    not isinstance(value, str)
    or len(value) != 64
    or any(char not in "0123456789abcdef" for char in value)
  ):
    _fail(f"{where} must be a lowercase 64-character SHA256 digest.")
  return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
  try:
    with path.open() as stream:
      payload = json.load(stream)
  except (OSError, json.JSONDecodeError) as exc:
    _fail(f"cannot read {label} artifact {path}: {exc}")
  return _mapping(payload, label)


def _resolve_reference(owner: Path, value: object, where: str) -> Path:
  if not isinstance(value, str) or not value:
    _fail(f"{where} must be a non-empty path string.")
  path = Path(value)
  if not path.is_absolute():
    path = owner.parent / path
  return path.resolve()


def _validate_protocol(protocol: dict[str, Any]) -> None:
  if protocol.get("id") != STRATIFICATION_PROTOCOL:
    _fail(f"protocol.id must be {STRATIFICATION_PROTOCOL!r}.")
  if _integer(protocol.get("version"), "protocol.version") != 1:
    _fail("protocol.version must be 1.")
  if _integer(protocol.get("trials"), "protocol.trials") != 5:
    _fail("strict v1 requires exactly 5 trials per clip.")
  if _number(protocol.get("threshold"), "protocol.threshold") != 0.8:
    _fail("strict v1 requires a mastery threshold of 0.8.")
  if _number(protocol.get("max_clip_seconds"), "protocol.max_clip_seconds") != 10.0:
    _fail("strict v1 requires a maximum clip duration of 10 seconds.")

  success = _mapping(protocol.get("success_criterion"), "success_criterion")
  if success.get("id") != "root_height_deviation":
    _fail("success_criterion.id must be 'root_height_deviation'.")
  if _number(success.get("tolerance_m"), "success_criterion.tolerance_m") != 0.2:
    _fail("success_criterion.tolerance_m must be 0.2.")

  randomness = _mapping(protocol.get("randomness"), "protocol.randomness")
  seed = _integer(randomness.get("seed"), "protocol.randomness.seed")
  if not 0 <= seed <= 2**31 - 1:
    _fail("protocol.randomness.seed must be in [0, 2^31 - 1].")
  if randomness.get("randomized_rollouts") is not True:
    _fail("protocol.randomness.randomized_rollouts must be true.")
  if randomness.get("sources") != list(_RANDOMNESS_SOURCES):
    _fail(f"protocol.randomness.sources must be {list(_RANDOMNESS_SOURCES)!r}.")
  if _integer(randomness.get("start_frame"), "randomness.start_frame") != 0:
    _fail("strict v1 fixes every rollout's start frame at zero.")
  if randomness.get("policy_actions") != "deterministic":
    _fail("protocol.randomness.policy_actions must be 'deterministic'.")
  if randomness.get("determinism") != "best_effort_mujoco_warp":
    _fail("protocol.randomness.determinism must disclose MuJoCo Warp best effort.")


def _validate_provenance(
  provenance: dict[str, Any], expected_paper_sha256: str
) -> None:
  expected = _sha256(expected_paper_sha256, "expected_paper_sha256")
  if expected != PAPER_SHA256:
    _fail(
      f"expected_paper_sha256 must remain pinned to {PAPER_ARXIV_ID} ({PAPER_SHA256})."
    )
  paper = _mapping(provenance.get("paper"), "provenance.paper")
  if paper.get("arxiv_id") != PAPER_ARXIV_ID:
    _fail(f"provenance.paper.arxiv_id must be {PAPER_ARXIV_ID!r}.")
  paper_sha256 = _sha256(paper.get("sha256"), "provenance.paper.sha256")
  if paper_sha256 != expected or provenance.get("paper_sha256") != expected:
    _fail("paper SHA256 is absent or does not match the pinned 2607.20110v1 PDF.")

  input_manifest = _mapping(
    provenance.get("input_manifest"), "provenance.input_manifest"
  )
  checkpoint = _mapping(provenance.get("base_checkpoint"), "provenance.base_checkpoint")
  for record, label, flat_key in (
    (input_manifest, "input_manifest", "input_manifest_sha256"),
    (checkpoint, "base_checkpoint", "base_checkpoint_sha256"),
  ):
    if not isinstance(record.get("path"), str) or not record["path"]:
      _fail(f"provenance.{label}.path must be a non-empty string.")
    digest = _sha256(record.get("sha256"), f"provenance.{label}.sha256")
    if provenance.get(flat_key) != digest:
      _fail(f"provenance.{flat_key} is absent or inconsistent.")
  declared = input_manifest.get("declared_provenance")
  if not isinstance(declared, dict) or not declared:
    _fail("provenance.input_manifest.declared_provenance must be non-empty.")
  if provenance.get("dataset_label") != "v1-proxy":
    _fail(
      "provenance.dataset_label must be 'v1-proxy'; exact paper data is not public."
    )


def _validate_envelope(
  payload: dict[str, Any], label: str, expected_paper_sha256: str
) -> None:
  if payload.get("schema") != STRATIFICATION_SCHEMA:
    _fail(f"{label}.schema must be {STRATIFICATION_SCHEMA!r}.")
  if _integer(payload.get("schema_version"), f"{label}.schema_version") != 1:
    _fail(f"{label}.schema_version must be 1.")
  if payload.get("kind") != _ARTIFACT_KINDS[label]:
    _fail(f"{label}.kind must be {_ARTIFACT_KINDS[label]!r}.")
  protocol = _mapping(payload.get("protocol"), f"{label}.protocol")
  provenance = _mapping(payload.get("provenance"), f"{label}.provenance")
  _validate_protocol(protocol)
  _validate_provenance(provenance, expected_paper_sha256)
  actual_set_sha256 = _sha256(
    payload.get("artifact_set_sha256"), f"{label}.artifact_set_sha256"
  )
  if not actual_set_sha256:
    _fail(f"{label}.artifact_set_sha256 is empty.")
  _mapping(payload.get("artifacts"), f"{label}.artifacts")


def _validate_clips(
  payload: dict[str, Any], owner: Path, label: str, *, require_nonempty: bool
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
  clips = payload.get("clips")
  if not isinstance(clips, list):
    _fail(f"{label}.clips must be a list.")
  if require_nonempty and not clips:
    _fail(f"{label}.clips must not be empty.")

  by_name: dict[str, dict[str, Any]] = {}
  for index, raw_clip in enumerate(clips):
    clip = _mapping(raw_clip, f"{label}.clips[{index}]")
    name = clip.get("name")
    if not isinstance(name, str) or not name:
      _fail(f"{label}.clips[{index}].name must be a non-empty string.")
    if name in by_name:
      _fail(f"{label} contains duplicate clip name {name!r}.")
    if not isinstance(clip.get("source"), str) or not clip["source"]:
      _fail(f"{label} clip {name!r} has no source label.")
    if not isinstance(clip.get("sequence_name"), str) or not clip["sequence_name"]:
      _fail(f"{label} clip {name!r} has no sequence_name.")
    _resolve_reference(owner, clip.get("path"), f"{label} clip {name!r}.path")

    start = _integer(clip.get("frame_start"), f"{label} clip {name!r}.frame_start")
    stop = _integer(clip.get("frame_stop"), f"{label} clip {name!r}.frame_stop")
    frames = _integer(clip.get("num_frames"), f"{label} clip {name!r}.num_frames")
    fps = _number(clip.get("fps"), f"{label} clip {name!r}.fps")
    if start < 0 or frames < 2 or stop != start + frames:
      _fail(
        f"{label} clip {name!r} has invalid half-open frame bounds "
        f"[{start}, {stop}) for num_frames={frames}."
      )
    if fps <= 0.0:
      _fail(f"{label} clip {name!r}.fps must be positive.")
    if frames / fps > STRATIFICATION_MAX_CLIP_SECONDS + 1e-12:
      _fail(f"{label} clip {name!r} exceeds the strict 10-second maximum.")

    trials = _integer(clip.get("trials"), f"{label} clip {name!r}.trials")
    if trials != STRATIFICATION_TRIALS:
      _fail(f"{label} clip {name!r} must record exactly 5 trials.")
    successes = _integer(
      clip.get("success_count"), f"{label} clip {name!r}.success_count"
    )
    if not 0 <= successes <= trials:
      _fail(f"{label} clip {name!r}.success_count is outside [0, {trials}].")
    success_rate = _number(
      clip.get("success_rate"), f"{label} clip {name!r}.success_rate"
    )
    if not math.isclose(success_rate, successes / trials, rel_tol=0.0, abs_tol=1e-12):
      _fail(f"{label} clip {name!r}.success_rate disagrees with its success count.")
    classification = clip.get("classification")
    if classification not in ("mastered", "challenging"):
      _fail(f"{label} clip {name!r}.classification is invalid.")
    if (success_rate >= STRATIFICATION_THRESHOLD) != (classification == "mastered"):
      _fail(f"{label} clip {name!r}.classification violates the 0.8 threshold.")
    by_name[name] = clip

  return clips, by_name


def _clip_identity(clip: dict[str, Any], owner: Path) -> tuple[object, ...]:
  return (
    clip["name"],
    clip["source"],
    clip["sequence_name"],
    _resolve_reference(owner, clip["path"], f"clip {clip['name']!r}.path"),
    clip["frame_start"],
    clip["frame_stop"],
    clip["num_frames"],
    float(clip["fps"]),
    clip["trials"],
    clip["success_count"],
    float(clip["success_rate"]),
    clip["classification"],
  )


def _validate_sequence_bounds(clips: list[dict[str, Any]]) -> None:
  sequences: dict[tuple[str, str], list[dict[str, Any]]] = {}
  for clip in clips:
    key = (clip["source"], clip["sequence_name"])
    sequences.setdefault(key, []).append(clip)
  for (source, sequence), parts in sequences.items():
    parts.sort(key=lambda clip: clip["frame_start"])
    for previous, current in zip(parts, parts[1:], strict=False):
      if previous["frame_stop"] != current["frame_start"]:
        _fail(
          f"sequence {source}/{sequence} has a gap or overlap between "
          f"{previous['name']!r} and {current['name']!r}."
        )


def _validate_report_record(
  name: str,
  record: dict[str, Any],
  clip: dict[str, Any],
  expected_classification: str,
) -> None:
  trials = _integer(record.get("trials"), f"report clip {name!r}.trials")
  if trials != STRATIFICATION_TRIALS:
    _fail(f"report clip {name!r} must record exactly 5 trials.")
  successes = _integer(
    record.get("success_count"), f"report clip {name!r}.success_count"
  )
  if not 0 <= successes <= trials:
    _fail(f"report clip {name!r}.success_count is outside [0, {trials}].")
  if record.get("successes") != successes:
    _fail(f"report clip {name!r}.successes disagrees with success_count.")
  success_rate = _number(
    record.get("success_rate"), f"report clip {name!r}.success_rate"
  )
  completion_rate = _number(
    record.get("completion_rate"), f"report clip {name!r}.completion_rate"
  )
  expected_rate = successes / trials
  if not math.isclose(success_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
    _fail(f"report clip {name!r}.success_rate disagrees with its success count.")
  if not math.isclose(completion_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12):
    _fail(f"report clip {name!r}.completion_rate disagrees with its success count.")

  if record.get("classification") != expected_classification:
    _fail(f"report clip {name!r} has the wrong mastered/challenging classification.")
  classified_as_mastered = success_rate >= STRATIFICATION_THRESHOLD
  if classified_as_mastered != (expected_classification == "mastered"):
    _fail(f"report clip {name!r} classification violates the 0.8 threshold.")

  for field in ("source", "sequence_name", "frame_start", "frame_stop", "num_frames"):
    if record.get(field) != clip[field]:
      _fail(f"report clip {name!r}.{field} disagrees with stratified.json.")
  if _number(record.get("fps"), f"report clip {name!r}.fps") != float(clip["fps"]):
    _fail(f"report clip {name!r}.fps disagrees with stratified.json.")
  for field in ("trials", "success_count", "success_rate", "classification"):
    if record.get(field) != clip[field]:
      _fail(f"report clip {name!r}.{field} disagrees with stratified.json.")
  duration = _number(
    record.get("duration_seconds"), f"report clip {name!r}.duration_seconds"
  )
  if not math.isclose(
    duration,
    clip["num_frames"] / float(clip["fps"]),
    rel_tol=0.0,
    abs_tol=1e-12,
  ):
    _fail(f"report clip {name!r}.duration_seconds disagrees with its frame bounds.")


def validate_stage2_manifests(
  stratified: str | Path,
  mastered: str | Path,
  challenging: str | Path,
  *,
  expected_paper_sha256: str = PAPER_SHA256,
) -> dict[str, Any]:
  """Validate and cross-check the strict v1 artifact set, returning its report.

  The report path is authenticated through the ``artifacts`` map present in each of
  the three supplied manifests.  Validation is intentionally fail closed: legacy
  clip-only manifests cannot start a strict Stage-II task.
  """
  paths = {
    "stratified": Path(stratified).resolve(),
    "mastered": Path(mastered).resolve(),
    "challenging": Path(challenging).resolve(),
  }
  payloads = {label: _load_json(path, label) for label, path in paths.items()}
  for label, payload in payloads.items():
    _validate_envelope(payload, label, expected_paper_sha256)

  report_ref = _mapping(
    payloads["stratified"].get("artifacts"), "stratified.artifacts"
  ).get("report")
  paths["report"] = _resolve_reference(
    paths["stratified"], report_ref, "stratified.artifacts.report"
  )
  payloads["report"] = _load_json(paths["report"], "report")
  _validate_envelope(payloads["report"], "report", expected_paper_sha256)

  canonical_protocol = payloads["stratified"]["protocol"]
  canonical_provenance = payloads["stratified"]["provenance"]
  canonical_set_sha256 = payloads["stratified"]["artifact_set_sha256"]
  for label, payload in payloads.items():
    if payload["protocol"] != canonical_protocol:
      _fail(f"{label}.protocol disagrees with stratified.json.")
    if payload["provenance"] != canonical_provenance:
      _fail(f"{label}.provenance disagrees with stratified.json.")
    if payload["artifact_set_sha256"] != canonical_set_sha256:
      _fail(f"{label}.artifact_set_sha256 disagrees with stratified.json.")
    artifact_refs = payload["artifacts"]
    for artifact_label, artifact_path in paths.items():
      referenced = _resolve_reference(
        paths[label],
        artifact_refs.get(artifact_label),
        f"{label}.artifacts.{artifact_label}",
      )
      if referenced != artifact_path:
        _fail(f"{label}.artifacts.{artifact_label} points to the wrong file.")

  stratified_clips, stratified_by_name = _validate_clips(
    payloads["stratified"], paths["stratified"], "stratified", require_nonempty=True
  )
  _, mastered_by_name = _validate_clips(
    payloads["mastered"], paths["mastered"], "mastered", require_nonempty=False
  )
  _, challenging_by_name = _validate_clips(
    payloads["challenging"],
    paths["challenging"],
    "challenging",
    require_nonempty=False,
  )
  _validate_sequence_bounds(stratified_clips)

  stratified_names = set(stratified_by_name)
  mastered_names = set(mastered_by_name)
  challenging_names = set(challenging_by_name)
  overlap = mastered_names & challenging_names
  if overlap:
    _fail(f"D_m and D_c overlap: {sorted(overlap)[:5]!r}.")
  if mastered_names | challenging_names != stratified_names:
    missing = stratified_names - (mastered_names | challenging_names)
    extra = (mastered_names | challenging_names) - stratified_names
    _fail(
      "D_m and D_c must be a disjoint cover of stratified clips; "
      f"missing={sorted(missing)[:5]!r}, extra={sorted(extra)[:5]!r}."
    )

  for subset_label, subset in (
    ("mastered", mastered_by_name),
    ("challenging", challenging_by_name),
  ):
    for name, clip in subset.items():
      if _clip_identity(clip, paths[subset_label]) != _clip_identity(
        stratified_by_name[name], paths["stratified"]
      ):
        _fail(f"{subset_label} clip {name!r} disagrees with stratified.json.")

  report_records = payloads["report"].get("clips")
  if not isinstance(report_records, dict):
    _fail("report.clips must be an object keyed by clip name.")
  if set(report_records) != stratified_names:
    _fail("report clip names must exactly match stratified.json.")
  for name, raw_record in report_records.items():
    record = _mapping(raw_record, f"report.clips[{name!r}]")
    classification = "mastered" if name in mastered_names else "challenging"
    _validate_report_record(name, record, stratified_by_name[name], classification)

  expected_set_sha256 = artifact_set_sha256(
    canonical_protocol,
    canonical_provenance,
    {label: payloads[label].get("clips") for label in _ARTIFACT_KINDS},
  )
  if canonical_set_sha256 != expected_set_sha256:
    _fail(
      "artifact_set_sha256 does not authenticate the exact stratified/D_m/D_c/report "
      "clip content."
    )

  return payloads["report"]


def validate_evaluation_manifest(
  manifest: str | Path,
  *,
  expected_paper_sha256: str = PAPER_SHA256,
) -> dict[str, Any]:
  """Validate that an evaluation manifest belongs to one strict v1 artifact set."""
  manifest_path = Path(manifest).resolve()
  payload = _load_json(manifest_path, "evaluation manifest")
  artifacts = _mapping(payload.get("artifacts"), "evaluation manifest.artifacts")
  paths = {
    label: _resolve_reference(
      manifest_path,
      artifacts.get(label),
      f"evaluation manifest.artifacts.{label}",
    )
    for label in ("stratified", "mastered", "challenging", "report")
  }
  member_labels = [
    label
    for label in ("stratified", "mastered", "challenging")
    if paths[label] == manifest_path
  ]
  if len(member_labels) != 1:
    _fail(
      "evaluation manifest must be exactly one of stratified.json, mastered.json, "
      "or challenging.json in its declared artifact set."
    )
  report = validate_stage2_manifests(
    paths["stratified"],
    paths["mastered"],
    paths["challenging"],
    expected_paper_sha256=expected_paper_sha256,
  )
  return {
    "validated": True,
    "paper_sha256": expected_paper_sha256,
    "artifact_set_sha256": report["artifact_set_sha256"],
    "manifest_kind": member_labels[0],
    "manifest_sha256": sha256_file(manifest_path),
    "report_sha256": sha256_file(paths["report"]),
  }
