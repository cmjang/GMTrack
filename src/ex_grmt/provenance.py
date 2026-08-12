"""Reproducibility metadata for the fixed Extreme-RGMT v1 protocol.

The paper does not publish every dataset and simulator parameter.  A run is therefore
only auditable when the exact public-paper revision, proxy inputs and dirty source
tree are recorded alongside its checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ex_grmt.protocol import PAPER_ARXIV_ID as PAPER_ID
from ex_grmt.protocol import PAPER_SHA256

DATA_PROTOCOL = "v1-proxy"
DISTILLATION_ACTION_SPACE = "deterministic_policy_mean_raw"
PUSH_INTERVAL_SOURCE = "2607.20110v1 Table II"
PUSH_MAGNITUDE_SOURCE = "InstinctLab proxy; 2607.20110v1 does not publish magnitude"


def sha256_file(path: str | Path) -> str:
  """Hash a required input without silently accepting a missing artifact."""
  resolved = Path(path).expanduser().resolve()
  digest = hashlib.sha256()
  with resolved.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_output(root: Path, *args: str) -> bytes:
  # The SIST compute image still ships a Git release predating `git -C`.
  # `cwd=` has identical repository-selection semantics and works there as well as
  # on current developer machines.
  result = subprocess.run(
    ("git", *args),
    check=True,
    cwd=root,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
  )
  return result.stdout


def source_state(root: str | Path) -> dict[str, Any]:
  """Return commit and hashes that distinguish a dirty working tree."""
  repo = Path(root).resolve()
  commit = _git_output(repo, "rev-parse", "HEAD").decode().strip()
  # `--porcelain` is the stable v1 format. Spell it without `=v1` because the
  # cluster's Git 1.8.3.1 supports the former but rejects the newer alias.
  status = _git_output(repo, "status", "--porcelain", "--untracked-files=all")
  diff = _git_output(repo, "diff", "--binary", "HEAD")
  tracked_digest = hashlib.sha256(diff).hexdigest()

  untracked: dict[str, str] = {}
  for raw in _git_output(
    repo, "ls-files", "--others", "--exclude-standard", "-z"
  ).split(b"\0"):
    if not raw:
      continue
    relative = raw.decode()
    if not relative.startswith(("src/", "scripts/", "tests/")):
      continue
    path = repo / relative
    if path.is_file():
      untracked[relative] = sha256_file(path)

  combined = hashlib.sha256()
  combined.update(commit.encode())
  combined.update(status)
  combined.update(diff)
  combined.update(json.dumps(untracked, sort_keys=True).encode())
  return {
    "commit": commit,
    "dirty": bool(status),
    "status_sha256": hashlib.sha256(status).hexdigest(),
    "tracked_diff_sha256": tracked_digest,
    "untracked_source_sha256": untracked,
    "source_tree_sha256": combined.hexdigest(),
  }


def _slurm_state() -> dict[str, str | None]:
  job_id = os.environ.get("SLURM_JOB_ID")
  batch_script = None
  if job_id is not None:
    # Compute-node batch shells on SIST do not always inherit Slurm's bin path.
    # Prefer PATH when it is configured, then use the cluster's stable install
    # location so provenance capture cannot abort an otherwise valid run.
    scontrol = shutil.which("scontrol") or "/opt/gridview/slurm/bin/scontrol"
    batch_script = subprocess.run(
      (scontrol, "write", "batch_script", job_id, "-"),
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    ).stdout
  return {
    "job_id": job_id,
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "batch_script": batch_script,
  }


def _jsonable(value: Any) -> Any:
  if is_dataclass(value):
    return _jsonable(asdict(value))
  if isinstance(value, dict):
    return {str(key): _jsonable(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  if value is None or isinstance(value, (str, int, float, bool)):
    return value
  return repr(value)


def build_run_provenance(
  *,
  repo_root: str | Path,
  train_cfg: dict[str, Any],
  manifests: dict[str, str | None],
  base_checkpoint: str | None,
  recovery_probability: float,
) -> dict[str, Any]:
  """Build the metadata written once when a training runner starts."""
  artifact_hashes: dict[str, dict[str, str]] = {}
  for name, value in manifests.items():
    if value is None:
      continue
    path = Path(value).expanduser().resolve()
    artifact_hashes[name] = {"path": str(path), "sha256": sha256_file(path)}
  if base_checkpoint is not None:
    path = Path(base_checkpoint).expanduser().resolve()
    artifact_hashes["base_checkpoint"] = {
      "path": str(path),
      "sha256": sha256_file(path),
    }

  return {
    "schema_version": 1,
    "paper": {"id": PAPER_ID, "sha256": PAPER_SHA256},
    "data_protocol": DATA_PROTOCOL,
    "assumptions": {
      "distillation_action_space": DISTILLATION_ACTION_SPACE,
      "push_interval_source": PUSH_INTERVAL_SOURCE,
      "push_magnitude_source": PUSH_MAGNITUDE_SOURCE,
      "recovery_probability": recovery_probability,
    },
    "source": source_state(repo_root),
    "artifacts": artifact_hashes,
    "slurm": _slurm_state(),
    "train_cfg": _jsonable(train_cfg),
  }


def write_run_provenance(path: str | Path, payload: dict[str, Any]) -> None:
  """Atomically write a run provenance JSON file."""
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(f"{destination.suffix}.tmp")
  try:
    with temporary.open("w") as stream:
      json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, destination)
  finally:
    temporary.unlink(missing_ok=True)
