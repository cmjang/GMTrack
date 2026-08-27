from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ex_grmt.provenance as provenance_module
from ex_grmt.provenance import (
  CHECKPOINT_OBSERVATION_SCHEMA_KEY,
  DATA_PROTOCOL,
  DISTILLATION_ACTION_SPACE,
  PAPER_ID,
  PAPER_SHA256,
  build_observation_schema,
  build_run_provenance,
  sha256_file,
  validate_checkpoint_observation_schema,
  write_run_provenance,
)


def _observation_schema(
  offsets: list[int], *, use_past_valid_mask: bool
) -> dict:
  actor_groups = ["proprio_hist", "action_hist", "command_window"]
  widths = {
    "proprio_hist": 640,
    "action_hist": 290,
    "command_window": 798,
    "critic": 261,
  }
  if use_past_valid_mask:
    actor_groups.append("past_valid_mask")
    widths["past_valid_mask"] = 21
  return build_observation_schema(
    actor_observation_groups=actor_groups,
    critic_observation_groups=["critic"],
    observation_group_widths=widths,
    command_window_offsets=offsets,
    critic_window_offsets=[],
    reconstruction_window_offsets=[],
    command_fps=50.0,
    command_token_dim=38,
    heading_closed_loop=False,
    history_length=10,
    proprio_term_names=[
      "projected_gravity",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
    ],
    proprio_term_dims=[3, 3, 29, 29],
    action_dim=29,
    use_past_valid_mask=use_past_valid_mask,
    use_history_valid_mask=False,
    use_future_valid_mask=False,
    use_reconstruction_valid_mask=False,
    command_position_encoding=(
      "sinusoidal_normalized_actual_offset"
      if use_past_valid_mask
      else "legacy_sinusoidal_slot_index"
    ),
    use_intent_aux=False,
    intent_latent_dim=64,
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
    observation_schema=_observation_schema(
      list(range(-10, 11)), use_past_valid_mask=False
    ),
  )

  assert payload["artifacts"]["motion_library"]["sha256"] == sha256_file(manifest)
  assert (
    payload["artifacts"]["base_checkpoint"]["sha256"]
    == hashlib.sha256(b"checkpoint").hexdigest()
  )
  assert payload["assumptions"]["recovery_probability"] == 0.0
  assert payload["source"]["commit"]
  assert payload["source"]["source_tree_sha256"]
  assert payload["observation_schema"]["schema"]["command"]["window_offsets"] == list(
    range(-10, 11)
  )

  destination = tmp_path / "run_provenance.json"
  write_run_provenance(destination, payload)
  assert json.loads(destination.read_text())["paper"]["sha256"] == PAPER_SHA256


def test_observation_schema_distinguishes_same_shape_temporal_layouts():
  baseline = _observation_schema(
    list(range(-10, 11)), use_past_valid_mask=False
  )
  causal = _observation_schema(
    list(range(-20, 1)), use_past_valid_mask=True
  )
  assert baseline["sha256"] != causal["sha256"]

  checkpoint = {CHECKPOINT_OBSERVATION_SCHEMA_KEY: causal}
  validate_checkpoint_observation_schema(
    checkpoint, causal, require_present=True
  )
  with pytest.raises(ValueError, match="does not match"):
    validate_checkpoint_observation_schema(
      checkpoint, baseline, require_present=False
    )


def test_causal_schema_fingerprints_critic_history_masks_and_auxiliary_layout():
  actor_offsets = [-32, -24, -16, -12, -8, -6, -4, -3, -2, -1, 0]
  critic_offsets = [1, 2, 4, 8, 16, 32, 64]
  reconstruction_offsets = [5, 10, 20]

  def build(critic: list[int]) -> dict:
    return build_observation_schema(
      actor_observation_groups=[
        "proprio_hist",
        "action_hist",
        "command_window",
        "history_valid_mask",
        "past_valid_mask",
      ],
      critic_observation_groups=[
        "proprio_hist",
        "action_hist",
        "history_valid_mask",
        "critic",
        "command_future_window",
        "future_valid_mask",
      ],
      observation_group_widths={
        "proprio_hist": 640,
        "action_hist": 290,
        "command_window": 418,
        "history_valid_mask": 10,
        "past_valid_mask": 11,
        "critic": 261,
        "command_future_window": 266,
        "future_valid_mask": 7,
        "future_reconstruction_target": 114,
        "future_reconstruction_valid_mask": 3,
      },
      command_window_offsets=actor_offsets,
      critic_window_offsets=critic,
      reconstruction_window_offsets=reconstruction_offsets,
      command_fps=50.0,
      command_token_dim=38,
      heading_closed_loop=False,
      history_length=10,
      proprio_term_names=[
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
      ],
      proprio_term_dims=[3, 3, 29, 29],
      action_dim=29,
      use_past_valid_mask=True,
      use_history_valid_mask=True,
      use_future_valid_mask=True,
      use_reconstruction_valid_mask=True,
      command_position_encoding="sinusoidal_normalized_actual_offset",
      use_intent_aux=True,
      intent_latent_dim=64,
    )

  schema = build(critic_offsets)
  payload = schema["schema"]
  assert payload["command"]["window_offsets"] == actor_offsets
  assert payload["command"]["critic_window_offsets"] == critic_offsets
  assert payload["command"]["reconstruction_window_offsets"] == (
    reconstruction_offsets
  )
  assert payload["command"]["position_encoding"] == (
    "sinusoidal_normalized_actual_offset"
  )
  assert payload["mask_layout"]["history_valid_mask"]["width"] == 10
  assert payload["mask_layout"]["future_valid_mask"]["width"] == 7
  assert payload["intent_auxiliary"]["latent_dim"] == 64
  assert schema["sha256"] != build([1, 2, 4, 8, 16, 32, 63])["sha256"]


def test_causal_schema_rejects_legacy_checkpoint_without_fingerprint():
  causal = _observation_schema(
    list(range(-20, 1)), use_past_valid_mask=True
  )
  with pytest.raises(ValueError, match="no observation-schema fingerprint"):
    validate_checkpoint_observation_schema({}, causal, require_present=True)

  baseline = _observation_schema(
    list(range(-10, 11)), use_past_valid_mask=False
  )
  validate_checkpoint_observation_schema({}, baseline, require_present=False)


def test_observation_schema_rejects_tampered_payload():
  schema = _observation_schema(
    list(range(-20, 1)), use_past_valid_mask=True
  )
  schema["schema"]["command"]["window_offsets"][0] = -19
  with pytest.raises(ValueError, match="stored SHA256"):
    validate_checkpoint_observation_schema(
      {CHECKPOINT_OBSERVATION_SCHEMA_KEY: schema},
      schema,
      require_present=True,
    )


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
