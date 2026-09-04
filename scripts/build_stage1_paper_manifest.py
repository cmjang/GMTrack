"""Build the Stage-I manifest from the configured motion sources.

Table IV gives the paper's Stage-I corpus as 3.096 h:
  LAFAN1 2.444 h (78.94%) | AMASS 0.511 h (16.51%) | in-house Xsens 0.141 h (4.55%)

Our LAFAN1 retarget (2.451 h) and the curated seed-simple set (0.510 h, standing in
for AMASS) already match the first two shares almost exactly. The in-house Xsens
recordings -- the paper's source of highly dynamic motion, and therefore of most of
the challenging set D_c that Stage-II acquisition trains on -- are not public. This
script substitutes an equally sized slice of airborne/rotational SEED motions.

Selection rule (deterministic, whole categories only):
  backflip_360, flip, tiger_jump_to_shoulder_roll_R  ->  0.139 h, 6 s short of 0.141
Alternative ``motiondecode-backflip`` profile:
  Back_Somersault, flip, tiger_jump_to_shoulder_roll_R, cartwheelin
  -> replaces the SEED backflips with all 18 pure MotionDecode takes while retaining
     approximately the same Table-IV duration.
Current ``final-backflip`` profile:
  the user-screened 14 MotionDecode + 16 SEED backflips, plus the SEED flip and
  tiger_jump_to_shoulder_roll_R categories. This is the active local proxy; the old
  profiles remain available only so historical runs can be reconstructed.
Excluded, with reasons:
  cartwheel_R   0.153 h  a single category already exceeds the 0.141 h budget
  safety_roll   0.043 h  ground rolls, not airborne
  side_roll_R   0.034 h  ground rolls, not airborne
  cartwheelin   0.010 h  only 2 unique clips
  ib_dodge*     0.019 h  combat micro-motions, not highly dynamic

Caveat recorded in the output: SEED ships mirrored duplicates (``_M``), so the 80
selected clips cover 52 unique motions. The duration matches Table IV -- which is
also a duration -- but the motion diversity of this slice is half its clip count.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SEED_HIGH_DYNAMIC_CATEGORIES = (
  "backflip_360",
  "flip",
  "tiger_jump_to_shoulder_roll_R",
)
MOTIONDECODE_BACKFLIP_CATEGORIES = (
  "back_somersault",
  "flip",
  "tiger_jump_to_shoulder_roll_R",
  # Four short clips (two original motions plus mirrors) close the duration gap
  # left when 18 MotionDecode back somersaults replace 28 longer SEED backflips.
  "cartwheelin",
)
FINAL_BACKFLIP_CATEGORIES = (
  "back_somersault",
  "backflip_360",
  "flip",
  "tiger_jump_to_shoulder_roll_R",
)
FPS = 50.0
_SEED_IDENTITY_RE = re.compile(r"^seed-(?:backflip|stunts)__(?P<identity>.+)$")


def _category(name: str, source: str) -> str:
  if source == "seed-backflip":
    return "backflip_360"
  if source == "motiondecode" and "Back_Somersault" in name:
    return "back_somersault"
  m = re.match(r"seed-stunts__([a-zA-Z_]+?)_\d", name)
  return m.group(1) if m else name


def _hours(clips: list[dict]) -> float:
  return sum(c["num_frames"] for c in clips) / FPS / 3600.0


def _deduplicate_seed_identities(clips: list[dict]) -> tuple[list[dict], list[dict]]:
  """Keep the first copy of a SEED action/actor identity across source manifests.

  ``seed_backflip`` and ``seed_stunts`` were exported from overlapping raw SEED
  directories.  Their manifest names have different source prefixes, so the normal
  duplicate-name check cannot detect that the underlying action/actor take is the
  same.  Input manifest order defines priority; the screened final-backflip manifest
  is passed first in the active build and therefore wins over the broad stunts pool.
  """
  seen: set[str] = set()
  kept: list[dict] = []
  removed: list[dict] = []
  for clip in clips:
    match = _SEED_IDENTITY_RE.fullmatch(clip["name"])
    if match is None:
      kept.append(clip)
      continue
    identity = match["identity"]
    if identity in seen:
      removed.append(clip)
      continue
    seen.add(identity)
    kept.append(clip)
  return kept, removed


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest-dir", default="logs/data_build/manifests")
  ap.add_argument("--base", default="stage1_lafan_seed_simple.json")
  ap.add_argument(
    "--high-dynamic",
    nargs="+",
    default=["backflip_final_selected.json", "seed_stunts_grounded.json"],
  )
  ap.add_argument(
    "--extension",
    nargs="*",
    default=[],
    help="Optional practical extension manifests appended after the Table-IV proxy.",
  )
  ap.add_argument(
    "--exclude-name-substring",
    action="append",
    default=[],
    help="Case-insensitive clip-name exclusion; repeat for multiple filters.",
  )
  ap.add_argument(
    "--deduplicate-seed-identities",
    action="store_true",
    help=(
      "Remove cross-manifest SEED copies with the same action/actor identity, "
      "preserving the first high-dynamic source's entry."
    ),
  )
  ap.add_argument(
    "--profile",
    choices=("seed", "motiondecode-backflip", "final-backflip"),
    default="final-backflip",
    help="High-dynamic proxy selection profile; defaults to the screened final set.",
  )
  ap.add_argument("--output", default="stage1_paper_mix_final_backflip_grounded.json")
  args = ap.parse_args()

  d = Path(args.manifest_dir)
  base = json.loads((d / args.base).read_text())
  base_clips = base["clips"]

  candidates: list[dict] = []
  for mf in args.high_dynamic:
    candidates += json.loads((d / mf).read_text())["clips"]

  categories = {
    "seed": SEED_HIGH_DYNAMIC_CATEGORIES,
    "motiondecode-backflip": MOTIONDECODE_BACKFLIP_CATEGORIES,
    "final-backflip": FINAL_BACKFLIP_CATEGORIES,
  }[args.profile]
  selected = [c for c in candidates if _category(c["name"], c["source"]) in categories]
  if not selected:
    raise ValueError(f"No clip matched {categories}.")
  duplicate_seed_clips: list[dict] = []
  if args.deduplicate_seed_identities:
    selected, duplicate_seed_clips = _deduplicate_seed_identities(selected)

  extension_clips: list[dict] = []
  for mf in args.extension:
    extension_clips += json.loads((d / mf).read_text())["clips"]

  names = {c["name"] for c in base_clips}
  added = selected + extension_clips
  clash = sorted(n for n in (c["name"] for c in added) if n in names)
  if clash:
    raise ValueError(f"Added clips already present in the base: {clash[:5]}")
  duplicate_added = sorted(
    n for n, count in Counter(c["name"] for c in added).items() if count > 1
  )
  if duplicate_added:
    raise ValueError(f"Duplicate clips across additions: {duplicate_added[:5]}")

  clips_before_filter = base_clips + added
  excluded_substrings = [value.casefold() for value in args.exclude_name_substring]
  excluded = [
    clip
    for clip in clips_before_filter
    if any(value in clip["name"].casefold() for value in excluded_substrings)
  ]
  clips = [clip for clip in clips_before_filter if clip not in excluded]
  paths = [d / c["path"] for c in clips]
  missing = [str(p) for p in paths if not p.resolve().exists()]
  if missing:
    raise FileNotFoundError(f"{len(missing)} npz missing, first: {missing[0]}")

  unique = len({c["name"].removesuffix("_M") for c in selected})
  payload = {
    "kind": base["kind"],
    "provenance": {
      "base": args.base,
      "profile": args.profile,
      "high_dynamic_sources": list(args.high_dynamic),
      "high_dynamic_categories": list(categories),
      "high_dynamic_hours": round(_hours(selected), 4),
      "high_dynamic_clips": len(selected),
      "high_dynamic_unique_after_unmirroring": unique,
      "seed_identity_deduplication": {
        "enabled": args.deduplicate_seed_identities,
        "removed_clips": len(duplicate_seed_clips),
        "removed_hours": round(_hours(duplicate_seed_clips), 4),
        "removed_names": [clip["name"] for clip in duplicate_seed_clips],
      },
      "extension_sources": list(args.extension),
      "extension_hours": round(_hours(extension_clips), 4),
      "extension_clips": len(extension_clips),
      "filter": {
        "excluded_name_substrings": list(args.exclude_name_substring),
        "excluded_clips": len(excluded),
        "excluded_hours": round(_hours(excluded), 4),
      },
      "total_hours_before_filter": round(_hours(clips_before_filter), 4),
      "total_hours": round(_hours(clips), 4),
      "paper_table_iv_total_hours": 3.096,
      "note": (
        "High-dynamic slice substitutes the paper's non-public in-house Xsens "
        "recordings (Table IV, 0.141 h). Sources are "
        + (
          "MotionDecode and SEED"
          if args.profile in {"motiondecode-backflip", "final-backflip"}
          else "SEED"
        )
        + ", not paper sources."
      ),
    },
    "clips": clips,
  }
  out = d / args.output
  out.write_text(json.dumps(payload, indent=2) + "\n")

  total = _hours(clips)
  print(f"wrote {out}  ({len(clips)} clips, {total:.4f} h; paper 3.096 h)")
  by_source: dict[str, list[dict]] = {}
  for c in clips:
    by_source.setdefault(c["source"], []).append(c)
  for src, cs in sorted(by_source.items(), key=lambda x: -_hours(x[1])):
    print(
      f"  {src:16s} {len(cs):4d} clips  {_hours(cs):.4f} h  {_hours(cs) / total * 100:5.2f}%"
    )
  print(
    f"  high-dynamic slice: {len(selected)} clips ({unique} unique), "
    f"{_hours(selected):.4f} h = {_hours(selected) / total * 100:.2f}% "
    "(paper Xsens share 4.55%)"
  )


if __name__ == "__main__":
  main()
