"""Tests for the packed multi-clip motion library and its difficulty sampler."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ex_grmt.mdp.motion_library import MotionLibrary
from ex_grmt.mdp.sampling import AdaptiveBinSampler

NUM_BODIES = 4
NUM_JOINTS = 5
FPS = 50.0


def _write_clip(path, num_frames: int, seed: int) -> None:
  rng = np.random.default_rng(seed)
  quat = rng.normal(size=(num_frames, NUM_BODIES, 4)).astype(np.float32)
  quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
  np.savez(
    path,
    fps=np.array([FPS]),
    joint_pos=rng.normal(size=(num_frames, NUM_JOINTS)).astype(np.float32),
    joint_vel=rng.normal(size=(num_frames, NUM_JOINTS)).astype(np.float32),
    body_pos_w=rng.normal(size=(num_frames, NUM_BODIES, 3)).astype(np.float32),
    body_quat_w=quat,
    body_lin_vel_w=rng.normal(size=(num_frames, NUM_BODIES, 3)).astype(np.float32),
    body_ang_vel_w=rng.normal(size=(num_frames, NUM_BODIES, 3)).astype(np.float32),
  )


@pytest.fixture
def library(tmp_path) -> MotionLibrary:
  lengths = [120, 75, 200]
  files = []
  for i, n in enumerate(lengths):
    p = tmp_path / f"clip{i}.npz"
    _write_clip(p, n, seed=i)
    files.append(p)
  return MotionLibrary(
    motion_files=files,
    body_indexes=torch.arange(NUM_BODIES),
    device="cpu",
  )


def test_packing_preserves_per_clip_content(library, tmp_path):
  assert library.num_clips == 3
  assert library.total_frames == 120 + 75 + 200
  assert library.clip_start.tolist() == [0, 120, 195]

  for i in range(3):
    raw = np.load(tmp_path / f"clip{i}.npz")
    start = int(library.clip_start[i])
    stop = start + int(library.clip_len[i])
    assert torch.allclose(
      library.joint_pos[start:stop], torch.as_tensor(raw["joint_pos"])
    )


def test_frame_index_is_not_clamped(library):
  """An out-of-range step must fail loudly, not silently read the next clip."""
  ids = torch.tensor([0])
  assert int(library.frame_index(ids, torch.tensor([0]))) == 0
  assert int(library.frame_index(ids, torch.tensor([119]))) == 119
  # Step 120 is past clip 0's end; the raw index lands in clip 1's territory, which
  # is exactly why the command term must never produce it.
  assert int(library.frame_index(ids, torch.tensor([120]))) == 120


def test_window_index_clamps_within_the_clip(library):
  offsets = torch.arange(-10, 11)
  # Clip 1 occupies rows [120, 195).
  idx = library.window_index(torch.tensor([1]), torch.tensor([0]), offsets)
  assert idx.shape == (1, 21)
  assert int(idx.min()) == 120, "window ran off the front of the clip"
  assert int(idx.max()) == 130

  idx = library.window_index(torch.tensor([1]), torch.tensor([74]), offsets)
  assert int(idx.max()) == 194, "window ran off the end of the clip"


def test_bins_are_one_second_wide(library):
  assert library.frames_per_bin == int(FPS)
  # 120 frames -> bins for [0,50), [50,100), [100,120) = 3
  assert library.clip_bins.tolist() == [3, 2, 4]
  assert library.max_bins == 4
  assert library.bin_mask[1].tolist() == [True, True, False, False]

  ids = torch.tensor([0, 0, 0])
  steps = torch.tensor([0, 60, 119])
  assert library.bin_of(ids, steps).tolist() == [0, 1, 2]


def test_missing_fps_raises(tmp_path):
  rng = np.random.default_rng(0)
  p = tmp_path / "no_fps.npz"
  np.savez(
    p,
    joint_pos=rng.normal(size=(10, NUM_JOINTS)).astype(np.float32),
    joint_vel=rng.normal(size=(10, NUM_JOINTS)).astype(np.float32),
    body_pos_w=rng.normal(size=(10, NUM_BODIES, 3)).astype(np.float32),
    body_quat_w=rng.normal(size=(10, NUM_BODIES, 4)).astype(np.float32),
    body_lin_vel_w=rng.normal(size=(10, NUM_BODIES, 3)).astype(np.float32),
    body_ang_vel_w=rng.normal(size=(10, NUM_BODIES, 3)).astype(np.float32),
  )
  with pytest.raises(KeyError, match="fps"):
    MotionLibrary([p], torch.arange(NUM_BODIES))


def test_missing_required_key_names_the_converter(tmp_path):
  p = tmp_path / "bad.npz"
  np.savez(p, fps=np.array([FPS]), joint_pos=np.zeros((10, NUM_JOINTS)))
  with pytest.raises(KeyError, match="csv_to_npz"):
    MotionLibrary([p], torch.arange(NUM_BODIES))


def test_manifest_roundtrip(library, tmp_path):
  manifest = tmp_path / "manifests" / "all.json"
  manifest.parent.mkdir()
  entries = [
    {
      "name": f"clip{i}",
      "source": "test",
      "path": f"../clip{i}.npz",
      "num_frames": int(library.clip_len[i]),
      "fps": FPS,
    }
    for i in range(3)
  ]
  manifest.write_text(json.dumps({"clips": entries}))

  full = MotionLibrary.from_manifest(manifest, torch.arange(NUM_BODIES))
  assert full.num_clips == 3

  subset = MotionLibrary.from_manifest(
    manifest, torch.arange(NUM_BODIES), subset=["clip0", "clip2"]
  )
  assert [c.name for c in subset.clips] == ["clip0", "clip2"]
  assert subset.total_frames == 320


##
# Adaptive bin sampler (Eq. 12-13)
##


def _sampler(library, clip_ids, **kw) -> AdaptiveBinSampler:
  return AdaptiveBinSampler(
    clip_ids=clip_ids,
    clip_bins=library.clip_bins[clip_ids],
    max_bins=library.max_bins,
    num_library_clips=library.num_clips,
    **kw,
  )


def test_sampler_never_draws_padding_bins(library):
  s = _sampler(library, torch.tensor([0, 1, 2]))
  assert float(s.probs[s.valid.logical_not()].sum()) == 0.0
  clips, bins = s.sample(500)
  assert torch.all(bins < library.clip_bins[clips])


def test_uniform_sampling_starts_uniform(library):
  s = _sampler(library, torch.tensor([0, 1, 2]))
  probs = s.probs[s.valid]
  assert torch.allclose(probs, torch.full_like(probs, 1.0 / s.num_valid_bins))
  h, _ = s.entropy_stats()
  assert h == pytest.approx(1.0, abs=1e-5)


def test_failures_shift_mass_toward_the_failing_bin(library):
  s = _sampler(library, torch.tensor([0, 1, 2]), alpha=0.5)
  before = float(s.probs[0, 1])
  for _ in range(5):
    s.record_failures(torch.tensor([0]), torch.tensor([1]))
    s.step_ema()
  assert float(s.probs[0, 1]) > before


def test_bin_weight_is_zero_outside_the_subset(library):
  """Consolidation clips must land in STAR's low-difficulty group E."""
  s = _sampler(library, torch.tensor([0]))  # only clip 0 is in this subset
  w = s.bin_weight(torch.tensor([0, 2]), torch.tensor([0, 0]))
  assert float(w[0]) > 0.0
  assert float(w[1]) == 0.0


def test_bin_weight_handles_clip_ids_beyond_the_subset_max(library):
  """Regression: sizing the reverse lookup by subset max would index out of bounds."""
  s = _sampler(library, torch.tensor([0]))
  # clip id 2 > max(subset) = 0
  s.bin_weight(torch.tensor([2]), torch.tensor([0]))
  s.flat_bin_id(torch.tensor([2]), torch.tensor([0]))


def test_uniform_ratio_must_be_positive(library):
  with pytest.raises(ValueError, match="uniform_ratio"):
    _sampler(library, torch.tensor([0]), uniform_ratio=0.0)
