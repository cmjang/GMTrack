from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import ex_grmt.provenance as provenance_module
from ex_grmt.provenance import (
  DATA_PROTOCOL,
  DISTILLATION_ACTION_SPACE,
  PAPER_ID,
  PAPER_SHA256,
  build_run_provenance,
  sha256_file,
  write_run_provenance,
)


def test_fixed_paper_and_action_semantics():
  assert PAPER_ID == "2607.20110v1"
  assert PAPER_SHA256 == (
    "55cca5c02f16c659e4ab3baf08d9ad1fb69865f37cfba084958ade0911cf51fe"
  )
  assert DATA_PROTOCOL == "v1-proxy"
  assert DISTILLATION_ACTION_SPACE == "deterministic_policy_mean_raw"
  paper = Path(__file__).resolve().parents[1] / "2607.20110v1.pdf"
  assert paper.exists(), "the pinned v1 PDF must travel with the reproduction"
  assert sha256_file(paper) == PAPER_SHA256


def test_run_provenance_hashes_every_training_input(tmp_path: Path):
  manifest = tmp_path / "manifest.json"
  checkpoint = tmp_path / "model.pt"
  manifest.write_text('{"clips": []}')
  checkpoint.write_bytes(b"checkpoint")

  payload = build_run_provenance(
    repo_root=Path(__file__).resolve().parents[1],
    train_cfg={"seed": 42, "algorithm": {"learning_rate": 1e-3}},
    manifests={"motion_library": str(manifest)},
    base_checkpoint=str(checkpoint),
    recovery_probability=0.0,
  )

  assert payload["artifacts"]["motion_library"]["sha256"] == sha256_file(manifest)
  assert (
    payload["artifacts"]["base_checkpoint"]["sha256"]
    == hashlib.sha256(b"checkpoint").hexdigest()
  )
  assert payload["assumptions"]["recovery_probability"] == 0.0
  assert payload["source"]["commit"]
  assert payload["source"]["source_tree_sha256"]

  destination = tmp_path / "run_provenance.json"
  write_run_provenance(destination, payload)
  assert json.loads(destination.read_text())["paper"]["sha256"] == PAPER_SHA256


def test_source_state_uses_git_1_8_compatible_invocations(tmp_path, monkeypatch):
  calls = []

  def run(command, **kwargs):
    calls.append((command, kwargs))
    stdout = b"deadbeef\n" if command[1:] == ("rev-parse", "HEAD") else b""
    return SimpleNamespace(stdout=stdout)

  monkeypatch.setattr(provenance_module.subprocess, "run", run)

  state = provenance_module.source_state(tmp_path)

  assert state["commit"] == "deadbeef"
  assert all(command[0] == "git" and "-C" not in command for command, _ in calls)
  assert all(kwargs["cwd"] == tmp_path.resolve() for _, kwargs in calls)
  status_command = next(command for command, _ in calls if command[1] == "status")
  assert "--porcelain" in status_command
  assert "--porcelain=v1" not in status_command


def test_slurm_state_falls_back_to_cluster_scontrol(monkeypatch):
  calls = []

  def run(command, **kwargs):
    calls.append((command, kwargs))
    return SimpleNamespace(stdout="#!/bin/bash\n")

  monkeypatch.setenv("SLURM_JOB_ID", "12345")
  monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
  monkeypatch.setattr(provenance_module.shutil, "which", lambda _: None)
  monkeypatch.setattr(provenance_module.subprocess, "run", run)

  state = provenance_module._slurm_state()

  assert calls[0][0] == (
    "/opt/gridview/slurm/bin/scontrol",
    "write",
    "batch_script",
    "12345",
    "-",
  )
  assert calls[0][1]["check"] is True
  assert state == {
    "job_id": "12345",
    "array_task_id": None,
    "batch_script": "#!/bin/bash\n",
  }
