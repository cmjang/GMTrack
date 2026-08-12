"""Build a compact, diversity-preserving SEED cartwheel manifest.

The SEED cartwheel export contains many actors performing the same named motion and
ships every take together with a mirrored ``_M`` copy.  Appending the complete set to
Stage I therefore makes clip count look like motion diversity and can let one action
family dominate the challenging set.  This curation keeps a target number of
representative ``cartwheel_R`` actor takes and always keeps each take's mirror.

The representative for a family is the take whose frame count is closest to that
family's median.  This avoids selecting a duration outlier while remaining fully
deterministic.  The source manifest and motion files are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_NAME_RE = re.compile(
  r"^seed-stunts__(?P<family>cartwheel_R_\d{3}|cartwheelin_\d{3})"
  r"__A(?P<actor>\d+)(?P<mirror>_M)?$"
)


def _source_sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_pairs(clips: list[dict[str, Any]]) -> dict[str, list[tuple[dict, dict]]]:
  grouped: dict[str, dict[str, dict[str, dict]]] = defaultdict(
    lambda: defaultdict(dict)
  )
  for clip in clips:
    match = _NAME_RE.fullmatch(clip["name"])
    if match is None:
      raise ValueError(f"Not a recognised SEED cartwheel clip: {clip['name']!r}")
    side = "mirror" if match["mirror"] else "original"
    actor_pair = grouped[match["family"]][match["actor"]]
    if side in actor_pair:
      raise ValueError(f"Duplicate {side} entry for {clip['name']!r}")
    actor_pair[side] = clip

  pairs: dict[str, list[tuple[dict, dict]]] = {}
  for family, actors in grouped.items():
    family_pairs = []
    for actor, pair in actors.items():
      if pair.keys() != {"original", "mirror"}:
        raise ValueError(f"Incomplete original/mirror pair for {family} actor A{actor}")
      original = pair["original"]
      mirror = pair["mirror"]
      if (original["num_frames"], original["fps"]) != (
        mirror["num_frames"],
        mirror["fps"],
      ):
        raise ValueError(f"Original/mirror metadata mismatch for {original['name']!r}")
      family_pairs.append((original, mirror))
    pairs[family] = family_pairs
  return pairs


def select_balanced_pairs(
  clips: list[dict[str, Any]], target_unique_takes: int = 10
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
  """Select ``cartwheel_R`` takes while covering all R action families.

  One median-duration take is selected from every R family first.  Remaining slots
  go to the most populated families, which adds actor variation where the source
  actually contains it without reintroducing the full take grid.  ``cartwheelin`` is
  deliberately excluded because it is a separate transition action rather than a
  ``cartwheel_R`` family.
  """
  pairs = _group_pairs(clips)
  eligible = {
    family: value for family, value in pairs.items() if "cartwheel_R_" in family
  }
  if target_unique_takes < len(eligible):
    raise ValueError(
      f"target_unique_takes={target_unique_takes} cannot cover all "
      f"{len(eligible)} cartwheel_R families"
    )
  if target_unique_takes > sum(len(value) for value in eligible.values()):
    raise ValueError("target_unique_takes exceeds available cartwheel_R pairs")

  ranked_by_family: dict[str, list[tuple[dict, dict]]] = {}
  for family, family_pairs in eligible.items():
    median_frames = statistics.median(pair[0]["num_frames"] for pair in family_pairs)
    ranked_by_family[family] = sorted(
      family_pairs,
      key=lambda pair: (
        abs(pair[0]["num_frames"] - median_frames),
        pair[0]["name"],
      ),
    )

  selected_pairs = [ranked_by_family[family][0] for family in sorted(eligible)]
  extras = [
    (family, rank, pair)
    for family, ranked in ranked_by_family.items()
    for rank, pair in enumerate(ranked[1:], start=1)
  ]
  extras.sort(
    key=lambda item: (
      item[1],
      -len(eligible[item[0]]),
      item[2][0]["name"],
    )
  )
  selected_pairs.extend(
    pair for _, _, pair in extras[: target_unique_takes - len(eligible)]
  )
  selected_pairs.sort(key=lambda pair: pair[0]["name"])

  selected: list[dict[str, Any]] = []
  selected_family_counts: dict[str, int] = defaultdict(int)
  for original, mirror in selected_pairs:
    family = _NAME_RE.fullmatch(original["name"])["family"]
    selected_family_counts[family] += 1
    selected.extend((original, mirror))
  source_family_counts = {
    family: len(value) for family, value in sorted(eligible.items())
  }
  return selected, source_family_counts, dict(selected_family_counts)


def build_balanced_manifest(
  source_manifest: str | Path,
  output_manifest: str | Path,
  *,
  target_unique_takes: int = 10,
) -> dict[str, Any]:
  """Build and write a relocatable complete-sequence manifest."""
  source = Path(source_manifest).resolve()
  output = Path(output_manifest).resolve()
  payload = json.loads(source.read_text())
  if payload.get("kind") != "complete_sequences":
    raise ValueError(f"{source} is not a complete-sequence manifest")

  selected, source_family_counts, selected_family_counts = select_balanced_pairs(
    payload["clips"], target_unique_takes
  )
  rebased: list[dict[str, Any]] = []
  for clip in selected:
    motion_path = (source.parent / clip["path"]).resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Missing selected motion file: {motion_path}")
    entry = dict(clip)
    entry["path"] = Path(os.path.relpath(motion_path, output.parent)).as_posix()
    rebased.append(entry)

  fps_values = {float(clip["fps"]) for clip in rebased}
  if len(fps_values) != 1:
    raise ValueError(f"Selected cartwheels have mixed frame rates: {fps_values}")
  fps = fps_values.pop()
  total_frames = sum(int(clip["num_frames"]) for clip in rebased)
  result = {
    "kind": "complete_sequences",
    "clip_count": len(rebased),
    "total_frames": total_frames,
    "total_seconds": total_frames / fps,
    "provenance": {
      "dataset": "seed-stunts-cartwheel-balanced",
      "source_manifest": source.name,
      "source_manifest_sha256": _source_sha256(source),
      "selection": (
        "cover every cartwheel_R action family with its median-duration actor "
        "take, allocate remaining slots to the most populated families, and keep "
        "each selected take's mirror; exclude cartwheelin"
      ),
      "target_unique_takes": target_unique_takes,
      "source_unique_takes_by_family": source_family_counts,
      "selected_unique_takes_by_family": selected_family_counts,
      "unique_after_unmirroring": len(rebased) // 2,
    },
    "clips": rebased,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  return result


def build_heldout_manifest(
  source_manifest: str | Path,
  selected_manifest: str | Path,
  output_manifest: str | Path,
) -> dict[str, Any]:
  """Build a name-disjoint test manifest from the unselected source clips."""
  source = Path(source_manifest).resolve()
  selected = Path(selected_manifest).resolve()
  output = Path(output_manifest).resolve()
  source_payload = json.loads(source.read_text())
  selected_payload = json.loads(selected.read_text())
  if source_payload.get("kind") != "complete_sequences":
    raise ValueError(f"{source} is not a complete-sequence manifest")
  if selected_payload.get("kind") != "complete_sequences":
    raise ValueError(f"{selected} is not a complete-sequence manifest")

  source_names = [clip["name"] for clip in source_payload["clips"]]
  selected_names = [clip["name"] for clip in selected_payload["clips"]]
  if len(source_names) != len(set(source_names)):
    raise ValueError("Source manifest contains duplicate clip names")
  if len(selected_names) != len(set(selected_names)):
    raise ValueError("Selected manifest contains duplicate clip names")
  missing = sorted(set(selected_names) - set(source_names))
  if missing:
    raise ValueError(f"Selected clips are absent from the source manifest: {missing}")

  selected_name_set = set(selected_names)
  heldout = [
    clip for clip in source_payload["clips"] if clip["name"] not in selected_name_set
  ]
  rebased: list[dict[str, Any]] = []
  for clip in heldout:
    motion_path = (source.parent / clip["path"]).resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"Missing held-out motion file: {motion_path}")
    entry = dict(clip)
    entry["path"] = Path(os.path.relpath(motion_path, output.parent)).as_posix()
    rebased.append(entry)

  fps_values = {float(clip["fps"]) for clip in rebased}
  if len(fps_values) != 1:
    raise ValueError(f"Held-out cartwheels have mixed frame rates: {fps_values}")
  fps = fps_values.pop()
  total_frames = sum(int(clip["num_frames"]) for clip in rebased)
  result = {
    "kind": "complete_sequences",
    "clip_count": len(rebased),
    "total_frames": total_frames,
    "total_seconds": total_frames / fps,
    "provenance": {
      "dataset": "seed-stunts-cartwheel-heldout-unseen",
      "source_manifest": source.name,
      "source_manifest_sha256": _source_sha256(source),
      "training_manifest": selected.name,
      "training_manifest_sha256": _source_sha256(selected),
      "selection": "source clips minus every name present in the training manifest",
      "source_clip_count": len(source_names),
      "excluded_training_clip_count": len(selected_names),
      "heldout_clip_count": len(rebased),
      "name_overlap_with_training": 0,
    },
    "clips": rebased,
  }
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2) + "\n")
  return result


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source-manifest",
    required=True,
  )
  parser.add_argument(
    "--output-manifest",
    default="data/current/seed_cartwheel_balanced_grounded.json",
  )
  parser.add_argument(
    "--heldout-manifest",
    default="data/test/seed_cartwheel_heldout_unseen92_grounded.json",
  )
  parser.add_argument("--target-unique-takes", type=int, default=10)
  return parser


def main() -> None:
  args = _parser().parse_args()
  result = build_balanced_manifest(
    args.source_manifest,
    args.output_manifest,
    target_unique_takes=args.target_unique_takes,
  )
  print(
    f"wrote {args.output_manifest}: {result['clip_count']} clips, "
    f"{result['provenance']['unique_after_unmirroring']} unique, "
    f"{result['total_seconds'] / 3600.0:.4f} h"
  )
  heldout = build_heldout_manifest(
    args.source_manifest,
    args.output_manifest,
    args.heldout_manifest,
  )
  print(
    f"wrote {args.heldout_manifest}: {heldout['clip_count']} clips, "
    f"{heldout['total_seconds'] / 3600.0:.4f} h, "
    "name_overlap_with_training=0"
  )


if __name__ == "__main__":
  main()
