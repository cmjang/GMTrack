"""Aggregate the five independent training seeds required by 2607.20110v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from ex_grmt.provenance import PAPER_SHA256

_METRICS = ("succ_pct", "mpjpe_mm", "d_vel", "d_acc")


@dataclass
class Config:
  inputs: tuple[str, ...]
  """Exactly five evaluation JSON files, one per independent training seed."""
  out: str


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
  output: dict[str, Any] = {
    "checkpoints": len(rows),
    "nonfinite_failures": sum(int(row.get("nonfinite_failures", 0)) for row in rows),
  }
  for key in _METRICS:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
      raise ValueError(f"Non-finite {key} in evaluation inputs: {values.tolist()}")
    output[key] = {
      "mean": float(values.mean()),
      "std": float(values.std(ddof=1)),
    }
  return output


def aggregate(paths: tuple[str, ...]) -> dict[str, Any]:
  if len(paths) != 5:
    raise ValueError(f"2607.20110v1 requires five training seeds, got {len(paths)}.")
  payloads = [json.loads(Path(path).read_text()) for path in paths]
  seeds = [payload.get("training_seed") for payload in payloads]
  if any(seed is None for seed in seeds) or len(set(seeds)) != 5:
    raise ValueError(
      f"Expected five distinct non-null training_seed values, got {seeds}."
    )
  for payload in payloads:
    paper = payload.get("paper", {})
    if paper.get("sha256") != PAPER_SHA256:
      raise ValueError("Evaluation input was not produced against pinned 2607.20110v1.")
    validation = payload.get("stratification_validation")
    if not isinstance(validation, dict) or validation.get("validated") is not True:
      raise ValueError("Evaluation input lacks strict v1 stratification validation.")
    if validation.get("paper_sha256") != PAPER_SHA256:
      raise ValueError("Evaluation stratification validation has the wrong paper hash.")
  manifest_hashes = {payload.get("manifest_sha256") for payload in payloads}
  artifact_set_hashes = {
    payload["stratification_validation"].get("artifact_set_sha256")
    for payload in payloads
  }
  modes = {payload.get("eval_mode") for payload in payloads}
  if len(manifest_hashes) != 1 or None in manifest_hashes:
    raise ValueError("Evaluation inputs do not use one identical manifest.")
  if len(artifact_set_hashes) != 1 or None in artifact_set_hashes:
    raise ValueError("Evaluation inputs do not use one validated stratification set.")
  if len(modes) != 1:
    raise ValueError("Evaluation inputs mix nominal and randomized modes.")

  source_sets = [set(payload.get("by_source", {})) for payload in payloads]
  if any(sources != source_sets[0] for sources in source_sets[1:]):
    raise ValueError("Evaluation inputs have different source groupings.")
  return {
    "schema_version": 1,
    "paper": payloads[0]["paper"],
    "manifest_sha256": manifest_hashes.pop(),
    "artifact_set_sha256": artifact_set_hashes.pop(),
    "eval_mode": modes.pop(),
    "training_seeds": sorted(int(seed) for seed in seeds),
    "overall": _aggregate([payload["overall"] for payload in payloads]),
    "by_source": {
      source: _aggregate([payload["by_source"][source] for payload in payloads])
      for source in sorted(source_sets[0])
    },
    "inputs": [str(Path(path).resolve()) for path in paths],
  }


def main(cfg: Config) -> None:
  result = aggregate(cfg.inputs)
  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  temporary = out.with_suffix(f"{out.suffix}.tmp")
  try:
    with temporary.open("w") as stream:
      json.dump(result, stream, indent=2, sort_keys=True)
    temporary.replace(out)
  finally:
    temporary.unlink(missing_ok=True)
  print(f"[ex-grmt] wrote five-seed aggregate to {out}")


if __name__ == "__main__":
  main(tyro.cli(Config))
