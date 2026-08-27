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
  assert library.clip_read_start.tolist() == library.clip_start.tolist()
  assert library.clip_read_len.tolist() == library.clip_len.tolist()

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

  _, valid = library.window_index_and_mask(
    torch.tensor([1]), torch.tensor([74]), offsets
  )
  assert valid.tolist() == [[True] * 11 + [False] * 10]


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


def test_manifest_ranges_slice_one_source_into_multiple_clips(tmp_path):
  source = tmp_path / "full_sequence.npz"
  _write_clip(source, 20, seed=123)
  manifest = tmp_path / "ranges.json"
  manifest.write_text(
    json.dumps(
      {
        "clips": [
          {
            "name": "first",
            "source": "test",
            "path": source.name,
            "frame_start": 2,
            "num_frames": 5,
            "fps": FPS,
          },
          {
            "name": "second",
            "source": "test",
            "path": source.name,
            "frame_start": 11,
            "num_frames": 4,
            "fps": FPS,
          },
        ]
      }
    )
  )

  lib = MotionLibrary.from_manifest(manifest, torch.tensor([1, 3]))
  raw = np.load(source)

  assert lib.clip_len.tolist() == [5, 4]
  assert lib.clip_start.tolist() == [2, 11]
  assert lib.clip_read_start.tolist() == [0, 0]
  assert lib.clip_read_len.tolist() == [20, 20]
  assert lib.total_frames == 20
  assert [info.frame_start for info in lib.clips] == [2, 11]
  assert torch.equal(lib.joint_pos, torch.as_tensor(raw["joint_pos"]))
  assert torch.equal(
    lib.body_pos_w, torch.as_tensor(raw["body_pos_w"][:, [1, 3]])
  )


def test_logical_fragment_window_reads_parent_context_and_masks_parent_ends(tmp_path):
  source = tmp_path / "full_sequence.npz"
  _write_clip(source, 20, seed=123)
  manifest = tmp_path / "fragment.json"
  manifest.write_text(
    json.dumps(
      {
        "clips": [
          {
            "name": "middle",
            "path": source.name,
            "frame_start": 5,
            "frame_stop": 10,
            "num_frames": 5,
            "fps": FPS,
          }
        ]
      }
    )
  )
  lib = MotionLibrary.from_manifest(manifest, torch.arange(NUM_BODIES))

  offsets = torch.tensor([-6, -5, -1, 0, 4, 5, 14, 15])
  idx, valid = lib.window_index_and_mask(
    torch.tensor([0]), torch.tensor([0]), offsets
  )

  assert idx.tolist() == [[0, 0, 4, 5, 9, 10, 19, 19]]
  assert valid.tolist() == [[False, True, True, True, True, True, True, False]]
  assert lib.window_index(torch.tensor([0]), torch.tensor([0]), offsets).tolist() == (
    idx.tolist()
  )


def test_nonadjacent_manifest_paths_load_once_and_preserve_order(tmp_path, monkeypatch):
  source_a = tmp_path / "sequence_a.npz"
  source_b = tmp_path / "sequence_b.npz"
  _write_clip(source_a, 20, seed=123)
  _write_clip(source_b, 20, seed=456)
  with np.load(source_a) as archive:
    joint_a = archive["joint_pos"].copy()
  with np.load(source_b) as archive:
    joint_b = archive["joint_pos"].copy()

  manifest = tmp_path / "ranges.json"
  manifest.write_text(
    json.dumps(
      {
        "clips": [
          {
            "name": "first",
            "path": "sequence_a.npz",
            "frame_start": 0,
            "num_frames": 5,
            "fps": FPS,
          },
          {
            "name": "middle",
            "path": "sequence_b.npz",
            "frame_start": 3,
            "num_frames": 4,
            "fps": FPS,
          },
          {
            "name": "last",
            "path": "./sequence_a.npz",
            "frame_start": 10,
            "num_frames": 5,
            "fps": FPS,
          },
        ]
      }
    )
  )

  real_load = np.load
  load_calls = 0
  key_reads: dict[str, int] = {}

  class CountingArchive:
    def __init__(self, path):
      self.archive = real_load(path)

    def __enter__(self):
      return self

    def __exit__(self, *_args):
      self.archive.close()

    def __contains__(self, key):
      return key in self.archive

    def __getitem__(self, key):
      key_reads[key] = key_reads.get(key, 0) + 1
      return self.archive[key]

  def counted_load(path):
    nonlocal load_calls
    load_calls += 1
    return CountingArchive(path)

  monkeypatch.setattr(np, "load", counted_load)
  monkeypatch.setattr(
    torch,
    "cat",
    lambda *_args, **_kwargs: pytest.fail(
      "manifest packing must copy directly into preallocated tensors"
    ),
  )
  lib = MotionLibrary.from_manifest(manifest, torch.arange(NUM_BODIES))

  assert load_calls == 2
  assert key_reads == {
    "joint_pos": 2,
    "joint_vel": 2,
    "body_pos_w": 2,
    "body_quat_w": 2,
    "body_lin_vel_w": 2,
    "body_ang_vel_w": 2,
    "fps": 2,
  }
  assert [info.name for info in lib.clips] == ["first", "middle", "last"]
  assert lib.total_frames == 40
  assert lib.clip_start.tolist() == [0, 23, 10]
  assert lib.clip_read_start.tolist() == [0, 20, 0]
  assert lib.clip_read_len.tolist() == [20, 20, 20]
  assert torch.equal(lib.joint_pos[:20], torch.as_tensor(joint_a))
  assert torch.equal(lib.joint_pos[20:], torch.as_tensor(joint_b))


@pytest.mark.parametrize(
  ("frame_start", "num_frames", "message"),
  [
    (-1, 3, "frame_start must be non-negative"),
    (8, 3, "exceeds the source length"),
    (0, 0, "num_frames must be positive"),
    (0, 1, "need at least 2"),
  ],
)
def test_manifest_rejects_invalid_frame_ranges(
  tmp_path, frame_start, num_frames, message
):
  source = tmp_path / "source.npz"
  _write_clip(source, 10, seed=0)
  manifest = tmp_path / "bad_range.json"
  manifest.write_text(
    json.dumps(
      {
        "clips": [
          {
            "name": "bad",
            "path": source.name,
            "frame_start": frame_start,
            "num_frames": num_frames,
            "fps": FPS,
          }
        ]
      }
    )
  )

  with pytest.raises(ValueError, match=message):
    MotionLibrary.from_manifest(manifest, torch.arange(NUM_BODIES))


@pytest.mark.parametrize("field", ["frame_start", "num_frames"])
def test_manifest_rejects_non_integer_frame_range_values(tmp_path, field):
  source = tmp_path / "source.npz"
  _write_clip(source, 10, seed=0)
  entry = {
    "name": "bad",
    "path": source.name,
    "frame_start": 0,
    "num_frames": 5,
    "fps": FPS,
  }
  entry[field] = 1.5
  manifest = tmp_path / "non_integer_range.json"
  manifest.write_text(json.dumps({"clips": [entry]}))

  with pytest.raises(TypeError, match=rf"{field} must be an integer"):
    MotionLibrary.from_manifest(manifest, torch.arange(NUM_BODIES))


def test_manifest_rejects_inconsistent_frame_stop(tmp_path):
  source = tmp_path / "source.npz"
  _write_clip(source, 10, seed=0)
  manifest = tmp_path / "bad_stop.json"
  manifest.write_text(
    json.dumps(
      {
        "clips": [
          {
            "name": "bad",
            "path": source.name,
            "frame_start": 2,
            "frame_stop": 8,
            "num_frames": 5,
            "fps": FPS,
          }
        ]
      }
    )
  )

  with pytest.raises(ValueError, match=r"frame_stop 8.*frame_start.*7"):
    MotionLibrary.from_manifest(manifest, torch.arange(NUM_BODIES))


def test_required_arrays_must_match_source_frame_count(tmp_path):
  source = tmp_path / "mismatched.npz"
  _write_clip(source, 10, seed=0)
  with np.load(source) as stored:
    arrays = {key: stored[key] for key in stored.files}
  arrays["body_ang_vel_w"] = arrays["body_ang_vel_w"][:-1]
  np.savez(source, **arrays)

  with pytest.raises(ValueError, match=r"body_ang_vel_w.*9 frames.*joint_pos.*10"):
    MotionLibrary([source], torch.arange(NUM_BODIES))


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
