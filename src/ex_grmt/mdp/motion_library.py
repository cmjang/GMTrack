"""Packed multi-clip motion library.

mjlab's ``MotionLoader`` (``mjlab/tasks/tracking/mdp/commands.py``) holds exactly one
motion clip. Extreme-RGMT trains a *generalist* tracker over a multi-source
distribution of thousands of clips (paper Table IV: 3.1 h across LAFAN1 / AMASS /
in-house Xsens), and Stage II additionally needs to sample disjoint subsets
(mastered vs challenging) per environment group.

This module stores every clip's frames concatenated along a single time axis plus
per-clip offsets, so lookups stay a single ``index_select`` regardless of clip count.
Only the *tracked* bodies are kept resident (mjlab's loader keeps both the full and
the indexed copy, which doubles VRAM for no benefit here).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REQUIRED_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _manifest_int(value: object, field: str, path: Path) -> int:
  """Read a manifest frame index/count without silently truncating floats."""
  if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
    raise TypeError(f"{path} manifest {field} must be an integer, got {value!r}.")
  return int(value)


@dataclass(frozen=True)
class ClipInfo:
  """Bookkeeping for a single clip inside the packed library."""

  name: str
  source: str
  path: str
  num_frames: int
  fps: float
  frame_start: int = 0
  frame_stop: int | None = None


def _npz_frame_count(path: Path) -> int:
  """Read the leading frame count from an npz member without decoding its payload."""
  member = "joint_pos.npy"
  with zipfile.ZipFile(path) as archive:
    if member not in archive.namelist():
      raise KeyError(
        f"{path} is missing required key 'joint_pos'. Motion npz files must be "
        f"produced by mjlab's csv_to_npz (body ordering is MuJoCo depth-first "
        f"and is NOT interchangeable with IsaacLab-produced files)."
      )
    with archive.open(member) as array_file:
      version = np.lib.format.read_magic(array_file)
      if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(array_file)
      elif version == (2, 0):
        shape, _, _ = np.lib.format.read_array_header_2_0(array_file)
      else:
        raise ValueError(
          f"{path} uses unsupported npy header version {version} for 'joint_pos'."
        )
  if not shape:
    raise ValueError(f"{path} required array 'joint_pos' has no frame dimension.")
  return int(shape[0])


def _load_motion_arrays(
  path: Path, *, load_fps: bool
) -> tuple[dict[str, np.ndarray], float | None]:
  """Load and validate one physical npz, decompressing each requested key once."""
  with np.load(path) as data:
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
      raise KeyError(
        f"{path} is missing required keys {missing}. Motion npz files must be "
        f"produced by mjlab's csv_to_npz (body ordering is MuJoCo depth-first "
        f"and is NOT interchangeable with IsaacLab-produced files)."
      )
    arrays = {key: data[key] for key in _REQUIRED_KEYS}
    if load_fps:
      if "fps" not in data:
        raise KeyError(
          f"{path} has no 'fps' key. Either regenerate it with mjlab's csv_to_npz "
          f"or pass clip_infos explicitly -- guessing the frame rate would silently "
          f"desynchronise the reference from the control loop."
        )
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    else:
      fps = None

  if arrays["joint_pos"].ndim == 0:
    raise ValueError(f"{path} required array 'joint_pos' has no frame dimension.")
  source_frames = int(arrays["joint_pos"].shape[0])
  for key, array in arrays.items():
    if array.ndim == 0:
      raise ValueError(f"{path} required array {key!r} has no frame dimension.")
    if int(array.shape[0]) != source_frames:
      raise ValueError(
        f"{path} required array {key!r} has {array.shape[0]} frames, but "
        f"'joint_pos' has {source_frames}."
      )
  return arrays, fps


def _selected_shapes(
  arrays: dict[str, np.ndarray], num_bodies: int
) -> dict[str, tuple[int, ...]]:
  """Shapes after selecting tracked bodies, excluding the frame dimension."""
  return {
    "joint_pos": arrays["joint_pos"].shape[1:],
    "joint_vel": arrays["joint_vel"].shape[1:],
    "body_pos_w": (num_bodies, *arrays["body_pos_w"].shape[2:]),
    "body_quat_w": (num_bodies, *arrays["body_quat_w"].shape[2:]),
    "body_lin_vel_w": (num_bodies, *arrays["body_lin_vel_w"].shape[2:]),
    "body_ang_vel_w": (num_bodies, *arrays["body_ang_vel_w"].shape[2:]),
  }


def _copy_range_into(
  packed: dict[str, torch.Tensor],
  arrays: dict[str, np.ndarray],
  source_slice: slice,
  destination_slice: slice,
  body_indexes: np.ndarray,
) -> None:
  """Copy one logical range from the current npz into its packed destination."""
  for key in ("joint_pos", "joint_vel"):
    packed[key][destination_slice].copy_(
      torch.as_tensor(arrays[key][source_slice], dtype=torch.float32)
    )
  for key in (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    packed[key][destination_slice].copy_(
      torch.as_tensor(arrays[key][source_slice, body_indexes], dtype=torch.float32)
    )


class MotionLibrary:
  """Many ``.npz`` motion clips packed into flat, contiguous tensors.

  Layout::

    joint_pos      (F, J)
    joint_vel      (F, J)
    body_pos_w     (F, B, 3)     B = len(body_indexes), tracked bodies only
    body_quat_w    (F, B, 4)     wxyz
    body_lin_vel_w (F, B, 3)
    body_ang_vel_w (F, B, 3)

  ``F`` is the sum of full parent-sequence lengths, with each physical npz packed
  exactly once. ``clip_start`` / ``clip_len`` describe the logical range used for
  sampling and termination, while ``clip_read_start`` / ``clip_read_len`` describe
  the parent range available to temporal windows. A ``(motion_id, local_step)`` pair
  maps to the global row ``clip_start[motion_id] + local_step``.

  Args:
    motion_files: Clip ``.npz`` paths, in the order they should be indexed.
    body_indexes: Row indices (into the npz body axis) of the tracked bodies.
      Must match the order of ``MotionCommandCfg.body_names``.
    device: Torch device for the packed tensors.
    clip_infos: Optional metadata parallel to ``motion_files``. Synthesised from the
      file names when omitted.
    bin_seconds: Width of one difficulty bin, in seconds of reference time. mjlab uses
      1 s bins (``bin_count = n_frames // (1 / step_dt) + 1``); we keep that default so
      the adaptive-sampling statistics stay comparable.
  """

  def __init__(
    self,
    motion_files: list[str | Path],
    body_indexes: torch.Tensor,
    device: str = "cpu",
    clip_infos: list[ClipInfo] | None = None,
    bin_seconds: float = 1.0,
  ) -> None:
    if not motion_files:
      raise ValueError("MotionLibrary requires at least one motion file.")
    if clip_infos is not None and len(clip_infos) != len(motion_files):
      raise ValueError(
        f"clip_infos has {len(clip_infos)} entries for {len(motion_files)} motion files."
      )

    self.device = device
    self._body_indexes = body_indexes.to(device)
    self.bin_seconds = bin_seconds

    infos: list[ClipInfo] = []
    lengths: list[int] = []
    body_idx_np = body_indexes.cpu().numpy()
    physical_groups: dict[Path, list[int]] = {}
    load_paths: dict[Path, Path] = {}
    logical_physical_paths: list[Path] = []
    frame_starts: list[int] = []
    source_lengths: dict[Path, int] = {}

    for i, path_arg in enumerate(motion_files):
      path = Path(path_arg)
      physical_path = path.resolve()
      physical_groups.setdefault(physical_path, []).append(i)
      load_paths.setdefault(physical_path, path)
      logical_physical_paths.append(physical_path)
      if physical_path not in source_lengths:
        source_lengths[physical_path] = _npz_frame_count(path)

      source_frames = source_lengths[physical_path]
      if clip_infos is None:
        frame_start = 0
        n = source_frames
      else:
        info = clip_infos[i]
        frame_start = _manifest_int(info.frame_start, "frame_start", path)
        n = _manifest_int(info.num_frames, "num_frames", path)
        if info.frame_stop is not None:
          frame_stop = _manifest_int(info.frame_stop, "frame_stop", path)
          if frame_stop != frame_start + n:
            raise ValueError(
              f"{path} frame_stop {frame_stop} does not equal frame_start + "
              f"num_frames ({frame_start + n})."
            )

      if frame_start < 0:
        raise ValueError(f"{path} frame_start must be non-negative, got {frame_start}.")
      if n <= 0:
        raise ValueError(f"{path} num_frames must be positive, got {n}.")
      if n < 2:
        raise ValueError(f"{path} logical clip has {n} frames; need at least 2.")
      if frame_start + n > source_frames:
        raise ValueError(
          f"{path} frame range [{frame_start}, {frame_start + n}) exceeds "
          f"the source length of {source_frames} frames."
        )
      frame_starts.append(frame_start)
      lengths.append(n)

    physical_starts: dict[Path, int] = {}
    packed_total_frames = 0
    for physical_path in physical_groups:
      physical_starts[physical_path] = packed_total_frames
      packed_total_frames += source_lengths[physical_path]

    packed: dict[str, torch.Tensor] = {}
    packed_shapes: dict[str, tuple[int, ...]] | None = None
    source_fps_by_path: dict[Path, float] = {}
    for physical_path, indices in physical_groups.items():
      path = load_paths[physical_path]
      arrays, source_fps = _load_motion_arrays(path, load_fps=True)
      source_frames = int(arrays["joint_pos"].shape[0])
      if source_frames != source_lengths[physical_path]:
        raise RuntimeError(
          f"{path} changed while being loaded: its header reported "
          f"{source_lengths[physical_path]} frames but its arrays contain "
          f"{source_frames}."
        )
      if source_fps is None:
        raise RuntimeError(f"Internal error: {path} fps was not loaded.")
      source_fps_by_path[physical_path] = source_fps
      if clip_infos is not None:
        for i in indices:
          if float(clip_infos[i].fps) != source_fps:
            raise ValueError(
              f"{path} manifest fps {clip_infos[i].fps} does not match the "
              f"physical motion fps {source_fps}."
            )

      selected_shapes = _selected_shapes(arrays, len(body_idx_np))
      if packed_shapes is None:
        packed_shapes = selected_shapes
        packed = {
          key: torch.empty((packed_total_frames, *shape), dtype=torch.float32)
          for key, shape in packed_shapes.items()
        }
      elif selected_shapes != packed_shapes:
        raise ValueError(
          f"{path} motion array shapes {selected_shapes} do not match the first "
          f"physical motion file's shapes {packed_shapes}."
        )

      destination_start = physical_starts[physical_path]
      _copy_range_into(
        packed,
        arrays,
        slice(0, source_frames),
        slice(destination_start, destination_start + source_frames),
        body_idx_np,
      )
      del arrays

    if packed_shapes is None:
      raise RuntimeError("Internal error: no physical motion files were packed.")

    if clip_infos is None:
      for path_arg, physical_path, n in zip(
        motion_files, logical_physical_paths, lengths, strict=True
      ):
        path = Path(path_arg)
        infos.append(
          ClipInfo(
            name=path.stem,
            source="unknown",
            path=str(path),
            num_frames=n,
            fps=source_fps_by_path[physical_path],
          )
        )
    else:
      infos = list(clip_infos)

    self.clips: list[ClipInfo] = infos
    self.num_clips = len(infos)

    self.joint_pos = packed["joint_pos"].to(device)
    self.joint_vel = packed["joint_vel"].to(device)
    self.body_pos_w = packed["body_pos_w"].to(device)
    self.body_quat_w = packed["body_quat_w"].to(device)
    self.body_lin_vel_w = packed["body_lin_vel_w"].to(device)
    self.body_ang_vel_w = packed["body_ang_vel_w"].to(device)

    self.clip_len = torch.tensor(lengths, dtype=torch.long, device=device)
    self.clip_start = torch.tensor(
      [
        physical_starts[physical_path] + frame_start
        for physical_path, frame_start in zip(
          logical_physical_paths, frame_starts, strict=True
        )
      ],
      dtype=torch.long,
      device=device,
    )
    self.clip_read_start = torch.tensor(
      [physical_starts[path] for path in logical_physical_paths],
      dtype=torch.long,
      device=device,
    )
    self.clip_read_len = torch.tensor(
      [source_lengths[path] for path in logical_physical_paths],
      dtype=torch.long,
      device=device,
    )
    self.total_frames = packed_total_frames

    fps_values = set(source_fps_by_path.values())
    if len(fps_values) > 1:
      raise ValueError(
        f"Clips have mixed frame rates {sorted(fps_values)}. Resample them to a "
        f"single fps before packing (the policy runs at a fixed control rate)."
      )
    self.fps = float(next(iter(fps_values)))

    # Difficulty bins, one row per clip, right-padded to the longest clip.
    frames_per_bin = max(int(round(self.bin_seconds * self.fps)), 1)
    self.frames_per_bin = frames_per_bin
    self.clip_bins = (
      torch.div(self.clip_len - 1, frames_per_bin, rounding_mode="floor") + 1
    )
    self.max_bins = int(self.clip_bins.max().item())
    # (num_clips, max_bins) True where the bin actually exists for that clip.
    self.bin_mask = (
      torch.arange(self.max_bins, device=device)[None, :] < self.clip_bins[:, None]
    )

  # -- indexing -------------------------------------------------------------

  def frame_index(
    self, motion_ids: torch.Tensor, local_steps: torch.Tensor
  ) -> torch.Tensor:
    """Map ``(motion_id, local_step)`` to a global row.

    Deliberately **not** clamped. An out-of-range local step is a bug in the command
    term's step/reset bookkeeping, and clamping would silently return a frame from
    the neighbouring clip in the packed buffer -- a corruption that would look like a
    plausible-but-wrong reference rather than an error. Callers that legitimately
    need clamping (the command window) use :meth:`window_index`.
    """
    return self.clip_start[motion_ids] + local_steps

  def window_index(
    self, motion_ids: torch.Tensor, local_steps: torch.Tensor, offsets: torch.Tensor
  ) -> torch.Tensor:
    """Global rows for a temporal window, clamped at physical-sequence endpoints.

    This compatibility wrapper discards the validity mask. New observation terms
    should call :meth:`window_index_and_mask` so a repeated endpoint cannot be
    mistaken for a genuinely stationary reference.
    """
    indices, _ = self.window_index_and_mask(motion_ids, local_steps, offsets)
    return indices

  def window_index_and_mask(
    self, motion_ids: torch.Tensor, local_steps: torch.Tensor, offsets: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Global rows and parent-bound validity for a temporal reference window.

    Args:
      motion_ids: ``(N,)`` clip id per environment.
      local_steps: ``(N,)`` current step within the clip.
      offsets: ``(W,)`` relative frame offsets, e.g. ``arange(-L, L + 1)``.

    Returns:
      A pair of ``(N, W)`` tensors: global row indices clamped to the physical
      parent sequence, and a boolean mask which is false only where an offset
      crossed a real parent-sequence endpoint. Logical fragment boundaries do not
      invalidate context; Stage-II fragments are views over their parent motion.
    """
    index = (
      self.clip_start[motion_ids][:, None]
      + local_steps[:, None]
      + offsets[None, :]
    )
    lower = self.clip_read_start[motion_ids][:, None]
    upper = lower + self.clip_read_len[motion_ids][:, None] - 1
    valid = (index >= lower) & (index <= upper)
    return torch.clamp(index, lower, upper), valid

  def bin_of(self, motion_ids: torch.Tensor, local_steps: torch.Tensor) -> torch.Tensor:
    """Difficulty-bin index within the clip for each env."""
    b = torch.div(local_steps, self.frames_per_bin, rounding_mode="floor")
    return torch.clamp(b, torch.zeros_like(b), self.clip_bins[motion_ids] - 1)

  def bin_start_step(
    self, motion_ids: torch.Tensor, bins: torch.Tensor
  ) -> torch.Tensor:
    """First local step belonging to ``bins`` (used when reseting into a bin)."""
    del motion_ids  # Bins are uniform width; kept for call-site symmetry.
    return bins * self.frames_per_bin

  # -- construction helpers -------------------------------------------------

  @classmethod
  def from_manifest(
    cls,
    manifest: str | Path,
    body_indexes: torch.Tensor,
    device: str = "cpu",
    subset: list[str] | None = None,
    bin_seconds: float = 1.0,
  ) -> MotionLibrary:
    """Build from a manifest written by ``ex_grmt.scripts.prepare_motions``.

    The manifest is ``{"clips": [{"name","source","path","num_frames","fps",\
    "frame_start","frame_stop"}, ...]}``. ``frame_start`` is optional and defaults
    to zero; together with ``num_frames`` it identifies a continuous logical clip
    within the referenced npz. When present, ``frame_stop`` must equal
    ``frame_start + num_frames``. Multiple entries may therefore reference disjoint
    ranges of the same full-sequence npz while sharing its readable parent context.
    ``subset`` optionally restricts to a set of clip names (order is preserved from
    the manifest, not from ``subset``).
    """
    manifest = Path(manifest)
    with manifest.open() as f:
      payload = json.load(f)

    entries = payload["clips"] if isinstance(payload, dict) else payload
    if subset is not None:
      wanted = set(subset)
      entries = [e for e in entries if e["name"] in wanted]
      if not entries:
        raise ValueError(
          f"No clips from subset ({len(wanted)} names) found in {manifest}."
        )

    root = manifest.parent
    infos, files = [], []
    for e in entries:
      path = Path(e["path"])
      if not path.is_absolute():
        # Manifest paths are relative to the manifest itself so the data dir stays
        # relocatable between the dev box and the cluster.
        path = (root / path).resolve()
      files.append(path)
      frame_stop = (
        _manifest_int(e["frame_stop"], "frame_stop", path)
        if "frame_stop" in e
        else None
      )
      infos.append(
        ClipInfo(
          name=e["name"],
          source=e.get("source", "unknown"),
          path=str(path),
          num_frames=_manifest_int(e["num_frames"], "num_frames", path),
          fps=float(e.get("fps", 50.0)),
          frame_start=_manifest_int(e.get("frame_start", 0), "frame_start", path),
          frame_stop=frame_stop,
        )
      )

    return cls(
      motion_files=files,
      body_indexes=body_indexes,
      device=device,
      clip_infos=infos,
      bin_seconds=bin_seconds,
    )

  def clip_ids_by_name(self, names: list[str]) -> torch.Tensor:
    """Resolve clip names to library ids, raising on anything unknown."""
    lookup = {info.name: i for i, info in enumerate(self.clips)}
    missing = [n for n in names if n not in lookup]
    if missing:
      raise KeyError(
        f"{len(missing)} clip name(s) not present in this library, e.g. {missing[:5]}"
      )
    return torch.tensor(
      [lookup[n] for n in names], dtype=torch.long, device=self.device
    )

  @property
  def clip_names(self) -> list[str]:
    """Human-readable names for every clip, in library order."""
    return [c.name for c in self.clips]

  def memory_bytes(self) -> int:
    """Resident size of the packed tensors, for logging / sanity checks."""
    return sum(
      t.numel() * t.element_size()
      for t in (
        self.joint_pos,
        self.joint_vel,
        self.body_pos_w,
        self.body_quat_w,
        self.body_lin_vel_w,
        self.body_ang_vel_w,
      )
    )

  def __repr__(self) -> str:
    return (
      f"MotionLibrary(clips={self.num_clips}, frames={self.total_frames}, "
      f"fps={self.fps:g}, duration={self.total_frames / self.fps / 60:.1f}min, "
      f"bodies={self.body_pos_w.shape[1]}, mem={self.memory_bytes() / 2**20:.0f}MiB)"
    )
