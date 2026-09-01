"""Tests for safe assembly of the active Stage-I manifest."""

import argparse
import json

import pytest

from gmtrack.scripts.curate_seed_simple import _parser, merge


def test_merge_defaults_use_complete_sequence_manifests():
  args = _parser().parse_args(["merge"])
  assert args.base_manifest == "logs/data_build/manifests/lafan1_full.json"
  assert args.additional_manifest == "logs/data_build/manifests/seed_simple_full.json"
  assert (
    args.output_manifest == "logs/data_build/manifests/stage1_lafan_seed_simple.json"
  )


def test_curate_default_writes_report_outside_manifest_directory():
  args = _parser().parse_args(["curate"])
  assert args.report == "logs/data_qc/seed_simple_selection.json"


def test_merge_rejects_a_legacy_presliced_manifest(tmp_path):
  base = tmp_path / "legacy.json"
  additional = tmp_path / "complete.json"
  output = tmp_path / "merged.json"
  base.write_text(json.dumps({"clips": []}))
  additional.write_text(json.dumps({"kind": "complete_sequences", "clips": []}))
  args = argparse.Namespace(
    base_manifest=str(base),
    additional_manifest=str(additional),
    output_manifest=str(output),
    base_source="lafan1",
  )

  with pytest.raises(ValueError, match="not a complete-sequence manifest"):
    merge(args)
  assert not output.exists()
