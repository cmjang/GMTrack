from __future__ import annotations

import json
from pathlib import Path

import pytest

from gmtrack.provenance import PAPER_ID, PAPER_SHA256
from gmtrack.scripts.aggregate_evaluations import aggregate


def _write_result(path: Path, seed: int, value: float) -> str:
  metrics = {
    "succ_pct": value,
    "mpjpe_mm": value + 1,
    "d_vel": value + 2,
    "d_acc": value + 3,
    "clips": 10,
  }
  path.write_text(
    json.dumps(
      {
        "training_seed": seed,
        "paper": {"id": PAPER_ID, "sha256": PAPER_SHA256},
        "manifest_sha256": "same",
        "stratification_validation": {
          "validated": True,
          "paper_sha256": PAPER_SHA256,
          "artifact_set_sha256": "artifact-set",
        },
        "eval_mode": "nominal",
        "overall": metrics,
        "by_source": {"proxy": metrics},
      }
    )
  )
  return str(path)


def test_aggregate_requires_and_reports_five_independent_seeds(tmp_path: Path):
  paths = tuple(
    _write_result(tmp_path / f"seed_{seed}.json", seed, float(seed - 42))
    for seed in range(42, 47)
  )
  result = aggregate(paths)
  assert result["training_seeds"] == [42, 43, 44, 45, 46]
  assert result["overall"]["succ_pct"] == pytest.approx({"mean": 2.0, "std": 2.5**0.5})


def test_aggregate_rejects_missing_or_duplicate_seeds(tmp_path: Path):
  four = tuple(
    _write_result(tmp_path / f"seed_{seed}.json", seed, 1.0) for seed in range(42, 46)
  )
  with pytest.raises(ValueError, match="five training seeds"):
    aggregate(four)

  duplicate = (*four, _write_result(tmp_path / "dup.json", 42, 1.0))
  with pytest.raises(ValueError, match="five distinct"):
    aggregate(duplicate)


def test_aggregate_rejects_unvalidated_evaluation(tmp_path: Path):
  paths = tuple(
    _write_result(tmp_path / f"seed_{seed}.json", seed, 1.0) for seed in range(42, 47)
  )
  payload = json.loads(Path(paths[0]).read_text())
  del payload["stratification_validation"]
  Path(paths[0]).write_text(json.dumps(payload))

  with pytest.raises(ValueError, match="lacks strict v1"):
    aggregate(paths)
