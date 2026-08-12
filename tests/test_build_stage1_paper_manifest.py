"""Tests for assembling the local Table-IV proxy manifest."""

from scripts.build_stage1_paper_manifest import _deduplicate_seed_identities


def _clip(name: str) -> dict:
  return {"name": name, "num_frames": 100, "fps": 50}


def test_seed_identity_deduplication_preserves_first_source_and_mirrors():
  screened = _clip("seed-backflip__flip_360_001__A304")
  mirror = _clip("seed-backflip__flip_360_001__A304_M")
  broad_duplicate = _clip("seed-stunts__flip_360_001__A304")
  distinct_actor = _clip("seed-stunts__flip_360_001__A305")

  kept, removed = _deduplicate_seed_identities(
    [screened, mirror, broad_duplicate, distinct_actor]
  )

  assert kept == [screened, mirror, distinct_actor]
  assert removed == [broad_duplicate]


def test_seed_identity_deduplication_does_not_merge_other_sources():
  motiondecode = _clip("motiondecode__flip_360_001__A304")
  seed = _clip("seed-stunts__flip_360_001__A304")

  kept, removed = _deduplicate_seed_identities([motiondecode, seed])

  assert kept == [motiondecode, seed]
  assert removed == []
