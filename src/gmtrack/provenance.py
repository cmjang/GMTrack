"""Reproducibility metadata for the fixed GMTrack v1 protocol.

A run is auditable when its protocol revision, inputs, and source-tree state are
recorded alongside its checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from gmtrack.protocol import PAPER_ARXIV_ID as PAPER_ID
from gmtrack.protocol import PAPER_SHA256

DATA_PROTOCOL = "v1-proxy"
DISTILLATION_ACTION_SPACE = "deterministic_policy_mean_raw"
PUSH_INTERVAL_SOURCE = "2607.20110v1 Table II"
PUSH_MAGNITUDE_SOURCE = "InstinctLab proxy; 2607.20110v1 does not publish magnitude"
OBSERVATION_SCHEMA_VERSION = 2
CHECKPOINT_OBSERVATION_SCHEMA_KEY = "gmtrack_observation_schema"
CHECKPOINT_OBSERVATION_SCHEMA_SUFFIX = "_observation_schema"


def sha256_file(path: str | Path) -> str:
  """Hash a required input without silently accepting a missing artifact."""
  resolved = Path(path).expanduser().resolve()
  digest = hashlib.sha256()
  with resolved.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _git_output(root: Path, *args: str) -> bytes:
  # `cwd=` keeps repository selection explicit without changing process state.
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
  # `--porcelain` is the stable v1 format.
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


def build_observation_schema(
  *,
  actor_observation_groups: list[str],
  critic_observation_groups: list[str],
  observation_group_widths: dict[str, int],
  command_window_offsets: list[int],
  critic_window_offsets: list[int],
  reconstruction_window_offsets: list[int],
  command_fps: float,
  command_token_dim: int,
  heading_closed_loop: bool,
  history_length: int,
  proprio_term_names: list[str],
  proprio_term_dims: list[int],
  action_dim: int,
  use_past_valid_mask: bool,
  use_history_valid_mask: bool,
  use_future_valid_mask: bool,
  use_reconstruction_valid_mask: bool,
  command_position_encoding: str,
  use_intent_aux: bool,
  intent_latent_dim: int,
  use_intent_in_actor: bool = False,
) -> dict[str, Any]:
  """Build a hash-stable policy observation ABI record.

  Tensor shapes alone cannot distinguish a symmetric reference window from a causal
  one. The payload therefore records temporal semantics, history packing, and mask
  alignment in addition to ordinary group widths.
  """
  mask_layout = (
    {
      "observation_group": "past_valid_mask",
      "width": len(command_window_offsets),
      "dtype": "bool",
      "ordering": "same_as_command_window_offsets",
      "true": "source frame is inside the physical parent sequence",
      "false": "token is clamped at a physical sequence endpoint",
    }
    if use_past_valid_mask
    else None
  )
  history_mask_layout = (
    {
      "observation_group": "history_valid_mask",
      "width": history_length,
      "dtype": "bool",
      "ordering": "oldest_to_current",
      "true": "slot is current or accumulated since episode reset",
      "false": "slot is synthetic observation-manager reset padding",
    }
    if use_history_valid_mask
    else None
  )
  future_mask_layout = (
    {
      "observation_group": "future_valid_mask",
      "width": len(critic_window_offsets),
      "dtype": "bool",
      "ordering": "same_as_critic_window_offsets",
      "true": "source frame is inside the physical parent sequence",
      "false": "token is clamped at the physical sequence end",
    }
    if use_future_valid_mask
    else None
  )
  reconstruction_mask_layout = (
    {
      "observation_group": "future_reconstruction_valid_mask",
      "width": len(reconstruction_window_offsets),
      "dtype": "bool",
      "ordering": "same_as_reconstruction_window_offsets",
      "true": "target frame is inside the physical parent sequence",
      "false": "target has no physical future frame and is excluded from MSE",
    }
    if use_reconstruction_valid_mask
    else None
  )
  intent_schema = {
    "enabled": use_intent_aux,
    "latent_distribution": "diagonal_gaussian" if use_intent_aux else None,
    "latent_dim": intent_latent_dim if use_intent_aux else None,
    "target_group": "future_reconstruction_target" if use_intent_aux else None,
    "target_offsets": reconstruction_window_offsets if use_intent_aux else [],
    "deployment": "omitted_from_onnx" if use_intent_aux else None,
  }
  # Preserve every existing version-2 fingerprint bit-for-bit. Only the new actor
  # family adds fields, so older causal checkpoints remain resumable and valid Stage-II
  # inputs under the unchanged schema version.
  if use_intent_in_actor:
    intent_schema.update(
      {
        "actor_conditioning": "deterministic_posterior_mean",
        "deployment": (
          "posterior_mean_exported; stochastic_sample_decoder_and_targets_omitted"
        ),
      }
    )

  schema = {
    "actor_observation_groups": actor_observation_groups,
    "critic_observation_groups": critic_observation_groups,
    "observation_group_widths": observation_group_widths,
    "command": {
      "window_offsets": command_window_offsets,
      "critic_window_offsets": critic_window_offsets,
      "reconstruction_window_offsets": reconstruction_window_offsets,
      "fps": command_fps,
      "offset_unit": "reference_frames_at_command_fps",
      "ordering": "ascending_oldest_to_newest",
      "token_dim": command_token_dim,
      "heading_closed_loop": heading_closed_loop,
      "position_encoding": command_position_encoding,
      "position_normalization": (
        "offset_divided_by_max_absolute_actor_offset"
        if command_position_encoding == "sinusoidal_normalized_actual_offset"
        else "none"
      ),
      "startup_fill": "clamp_to_physical_parent_sequence_boundary",
      "logical_fragment_window_context": "read_from_complete_parent_sequence",
    },
    "history": {
      "length": history_length,
      "proprio_terms": [
        {"name": name, "width": width}
        for name, width in zip(proprio_term_names, proprio_term_dims, strict=True)
      ],
      "proprio_flattening": "term_major_then_time_major_oldest_to_newest",
      "action_dim": action_dim,
      "action_flattening": "time_major_oldest_to_newest",
    },
    "mask_layout": {
      "history_valid_mask": history_mask_layout,
      "past_valid_mask": mask_layout,
      "future_valid_mask": future_mask_layout,
      "future_reconstruction_valid_mask": reconstruction_mask_layout,
    },
    "intent_auxiliary": intent_schema,
  }
  encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
  return {
    "version": OBSERVATION_SCHEMA_VERSION,
    "sha256": hashlib.sha256(encoded).hexdigest(),
    "schema": schema,
  }


def validate_checkpoint_observation_schema(
  checkpoint: dict[str, Any],
  expected: dict[str, Any],
  *,
  require_present: bool,
) -> None:
  """Reject checkpoints whose observation timing/layout differs from the runtime.

  Legacy checkpoints without a fingerprint remain loadable only by the exact
  registered legacy observation ABI. Non-default and causal layouts require the
  field because shared token encoders do not carry window timing in parameter shapes.
  Pre-rename checkpoints carry the fingerprint under the old project key and are
  validated exactly as strictly as current ones.
  """
  legacy_keys = sorted(
    key
    for key in checkpoint
    if isinstance(key, str)
    and key != CHECKPOINT_OBSERVATION_SCHEMA_KEY
    and key.endswith(CHECKPOINT_OBSERVATION_SCHEMA_SUFFIX)
  )
  if CHECKPOINT_OBSERVATION_SCHEMA_KEY in checkpoint:
    key = CHECKPOINT_OBSERVATION_SCHEMA_KEY
  elif len(legacy_keys) == 1:
    key = legacy_keys[0]
  elif len(legacy_keys) > 1:
    raise ValueError("Checkpoint has multiple legacy observation-schema fingerprints.")
  else:
    key = None
  if key is None:
    if require_present:
      raise ValueError(
        "Checkpoint has no observation-schema fingerprint; it cannot be loaded by "
        "a non-legacy task whose tensor shapes do not encode observation timing."
      )
    return

  saved = checkpoint[key]
  if not isinstance(saved, dict):
    raise TypeError("Checkpoint observation schema must be a mapping.")
  if set(saved) != {"version", "sha256", "schema"}:
    raise ValueError(
      "Checkpoint observation schema must contain exactly version, sha256, and schema."
    )
  if saved["version"] != OBSERVATION_SCHEMA_VERSION:
    raise ValueError(f"Unsupported observation schema version {saved['version']!r}.")
  encoded = json.dumps(saved["schema"], sort_keys=True, separators=(",", ":")).encode()
  actual_hash = hashlib.sha256(encoded).hexdigest()
  if actual_hash != saved["sha256"]:
    raise ValueError(
      "Checkpoint observation schema payload does not match its stored SHA256."
    )
  if saved["sha256"] != expected["sha256"]:
    raise ValueError(
      "Checkpoint observation schema does not match the current task: "
      f"expected {expected['sha256']}, got {saved['sha256']}."
    )


def build_run_provenance(
  *,
  repo_root: str | Path,
  train_cfg: dict[str, Any],
  manifests: dict[str, str | None],
  base_checkpoint: str | None,
  recovery_probability: float,
  observation_schema: dict[str, Any],
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
    "observation_schema": observation_schema,
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
