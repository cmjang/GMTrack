"""Migrate local manifests from legacy ``data/motions_*`` paths to ``data/datasets``.

The migration is semantic-preserving: NPZ bytes and clip metadata do not change.
Strict Stage-II bundles authenticate clip paths, so their shared artifact hashes and
source-manifest provenance must be regenerated as one atomic set after the rewrite.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

from ex_grmt.protocol import artifact_set_sha256, sha256_file, validate_stage2_manifests

FORMAL_SOURCE = (
  "stage1_paper_mix_final_backflip_cartwheel_balanced_grounded_no_fall_getup.json"
)
PATH_REWRITES = {
  "motions_stage1_full/": "datasets/stage1_full/",
  "motions_motiondecode_backflip_grounded/": "datasets/motiondecode_backflip_grounded/",
  "motions_seed_backflip_grounded/": "datasets/seed_backflip_grounded/",
  "motions_seed_stunts_grounded/": "datasets/seed_stunts_grounded/",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  temporary = path.with_suffix(f"{path.suffix}.tmp")
  with temporary.open("w") as stream:
    json.dump(payload, stream, indent=2, allow_nan=False)
  os.replace(temporary, path)


def _rewrite_clip_paths(payload: dict[str, Any]) -> bool:
  clips = payload.get("clips")
  if not isinstance(clips, list):
    return False
  changed = False
  for clip in clips:
    if not isinstance(clip, dict) or not isinstance(clip.get("path"), str):
      continue
    rewritten = clip["path"]
    for old, new in PATH_REWRITES.items():
      rewritten = rewritten.replace(old, new)
    if rewritten != clip["path"]:
      clip["path"] = rewritten
      changed = True
  return changed


def _rewrite_provenance_paths(payload: dict[str, Any]) -> bool:
  provenance = payload.get("provenance")
  if not isinstance(provenance, dict):
    return False
  changed = False
  for key, value in provenance.items():
    if isinstance(value, str) and "/data/raw/" in value:
      provenance[key] = value.replace("/data/raw/", "/data/datasets/raw/")
      changed = True
  return changed


def migrate(data_dir: Path) -> dict[str, str]:
  data_dir = data_dir.resolve()
  formal = data_dir / "current" / FORMAL_SOURCE
  formal_before = json.loads(formal.read_text())
  previous_source_sha = (
    formal_before.get("provenance", {})
    .get("path_layout_migration", {})
    .get("previous_manifest_sha256", sha256_file(formal))
  )

  for root_name in ("current", "test"):
    for path in sorted((data_dir / root_name).rglob("*.json")):
      payload = json.loads(path.read_text())
      clips_changed = _rewrite_clip_paths(payload)
      provenance_changed = _rewrite_provenance_paths(payload)
      if clips_changed or provenance_changed:
        if path == formal:
          payload.setdefault("provenance", {})["path_layout_migration"] = {
            "id": "data-datasets-v1",
            "semantic_content_unchanged": True,
            "previous_manifest_sha256": previous_source_sha,
            "motion_payloads_changed": False,
          }
        _write_json(path, payload)

  source_sha = sha256_file(formal)
  migration = {
    "id": "data-datasets-v1",
    "semantic_content_unchanged": True,
    "previous_input_manifest_sha256": previous_source_sha,
    "motion_payloads_changed": False,
  }
  artifact_hashes: dict[str, str] = {}
  for stratified_path in sorted((data_dir / "current").rglob("stratified.json")):
    bundle_dir = stratified_path.parent
    paths = {
      "stratified": stratified_path,
      "mastered": bundle_dir / "mastered.json",
      "challenging": bundle_dir / "challenging.json",
      "report": bundle_dir / "stratification_report.json",
    }
    payloads = {label: json.loads(path.read_text()) for label, path in paths.items()}
    provenance = copy.deepcopy(payloads["stratified"]["provenance"])
    provenance["input_manifest"]["path"] = f"data/current/{FORMAL_SOURCE}"
    provenance["input_manifest"]["sha256"] = source_sha
    provenance["input_manifest_sha256"] = source_sha
    provenance["layout_migration"] = migration

    artifact_hash = artifact_set_sha256(
      payloads["stratified"]["protocol"],
      provenance,
      {label: payloads[label]["clips"] for label in paths},
    )
    for label, path in paths.items():
      payloads[label]["provenance"] = provenance
      payloads[label]["artifact_set_sha256"] = artifact_hash
      _write_json(path, payloads[label])

    validate_stage2_manifests(
      paths["stratified"], paths["mastered"], paths["challenging"]
    )
    artifact_hashes[bundle_dir.name] = artifact_hash

    heading_contract = bundle_dir / "heading_contract.json"
    if heading_contract.is_file():
      contract = json.loads(heading_contract.read_text())
      contract["source_manifest_sha256"] = source_sha
      contract["artifact_set_sha256"] = artifact_hash
      contract["layout_migration"] = migration
      _write_json(heading_contract, contract)

  return {"source_manifest_sha256": source_sha, **artifact_hashes}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", type=Path, default=Path("data"))
  result = migrate(parser.parse_args().data_dir)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
