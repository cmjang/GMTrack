"""Fail closed on the short Stage-II trust-region health gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class Config:
  run_dir: str
  out: str | None = None
  expected_iterations: int = 500
  desired_kl: float = 0.01
  lr_floor: float = 1.0e-5
  max_floor_kl_streak: int = 100


def _series(run_dir: Path, tag: str) -> dict[int, float]:
  values: dict[int, float] = {}
  event_files = sorted(run_dir.glob("events.out.tfevents.*"))
  if not event_files:
    raise ValueError(f"No TensorBoard event files in {run_dir}.")
  for event_file in event_files:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", ()):
      continue
    for event in accumulator.Scalars(tag):
      values[int(event.step)] = float(event.value)
  if not values:
    raise ValueError(f"Missing required TensorBoard scalar {tag!r} in {run_dir}.")
  return values


def _max_true_streak(flags: list[bool]) -> int:
  longest = current = 0
  for flag in flags:
    current = current + 1 if flag else 0
    longest = max(longest, current)
  return longest


def check_gate(cfg: Config) -> dict:
  run_dir = Path(cfg.run_dir).resolve()
  kl = _series(run_dir, "Loss/kl")
  lr = _series(run_dir, "Loss/learning_rate")
  actor_grad = _series(run_dir, "Loss/actor_grad_norm")
  critic_grad = _series(run_dir, "Loss/critic_grad_norm")
  initial_mean = _series(run_dir, "Loss/initial_deterministic_mean_max_abs_diff")
  initial_kl = _series(run_dir, "Loss/initial_reference_kl_max")
  failure_explicit = _series(run_dir, "Loss/tracking_failure_explicit_steps")
  failure_derived = _series(run_dir, "Loss/tracking_failure_derived_steps")
  failure_missing = _series(run_dir, "Loss/tracking_failure_missing_steps")
  nonfinite = _series(run_dir, "Episode_Termination/nonfinite_physics_state")
  common_steps = sorted(
    set(kl)
    & set(lr)
    & set(actor_grad)
    & set(critic_grad)
    & set(failure_explicit)
    & set(failure_derived)
    & set(failure_missing)
    & set(nonfinite)
  )
  if not common_steps or common_steps[-1] < cfg.expected_iterations - 1:
    raise ValueError(
      f"Gate is incomplete: last common iteration is "
      f"{common_steps[-1] if common_steps else None}, expected at least "
      f"{cfg.expected_iterations - 1}."
    )

  finite = all(
    math.isfinite(series[step])
    for series in (kl, lr, actor_grad, critic_grad)
    for step in common_steps
  )
  bad_floor_streak = _max_true_streak(
    [
      lr[step] <= cfg.lr_floor * (1.0 + 1.0e-6) and kl[step] > 2.0 * cfg.desired_kl
      for step in common_steps
    ]
  )
  reference_ok = (
    max(initial_mean.values()) <= 1.0e-7 and max(initial_kl.values()) <= 1.0e-7
  )
  tracking_failure_hook_ok = (
    min(failure_explicit[step] for step in common_steps) > 0.0
    and max(failure_derived[step] for step in common_steps) == 0.0
    and max(failure_missing[step] for step in common_steps) == 0.0
  )
  nonfinite_terminations = max(nonfinite[step] for step in common_steps)

  passed = (
    finite
    and reference_ok
    and tracking_failure_hook_ok
    and nonfinite_terminations == 0.0
    and bad_floor_streak < cfg.max_floor_kl_streak
  )
  return {
    "schema_version": 1,
    "run_dir": str(run_dir),
    "passed": passed,
    "criteria": {
      "expected_iterations": cfg.expected_iterations,
      "desired_kl": cfg.desired_kl,
      "lr_floor": cfg.lr_floor,
      "max_floor_kl_streak": cfg.max_floor_kl_streak,
    },
    "observed": {
      "last_iteration": common_steps[-1],
      "all_core_scalars_finite": finite,
      "initial_reference_ok": reference_ok,
      "initial_mean_max_abs_diff": max(initial_mean.values()),
      "initial_reference_kl_max": max(initial_kl.values()),
      "tracking_failure_hook_ok": tracking_failure_hook_ok,
      "tracking_failure_explicit_steps_min": min(
        failure_explicit[step] for step in common_steps
      ),
      "tracking_failure_derived_steps_max": max(
        failure_derived[step] for step in common_steps
      ),
      "tracking_failure_missing_steps_max": max(
        failure_missing[step] for step in common_steps
      ),
      "floor_high_kl_streak": bad_floor_streak,
      "kl_last": kl[common_steps[-1]],
      "lr_last": lr[common_steps[-1]],
      "nonfinite_terminations_max": nonfinite_terminations,
    },
  }


def main(cfg: Config) -> None:
  result = check_gate(cfg)
  out = Path(cfg.out) if cfg.out is not None else Path(cfg.run_dir) / "gate_report.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  temporary = out.with_suffix(f"{out.suffix}.tmp")
  try:
    with temporary.open("w") as stream:
      json.dump(result, stream, indent=2, sort_keys=True)
    temporary.replace(out)
  finally:
    temporary.unlink(missing_ok=True)
  print(json.dumps(result, indent=2, sort_keys=True))
  if not result["passed"]:
    raise SystemExit(2)


if __name__ == "__main__":
  main(tyro.cli(Config))
