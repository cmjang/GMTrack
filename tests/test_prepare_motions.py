"""Tests for the clip slicing rules (paper Sec. IV-C)."""

from __future__ import annotations

import mjlab
import tyro

from ex_grmt.scripts.prepare_motions import Config, _slice_ranges

ROWS_PER_CLIP = 300  # 10 s at 30 fps
MIN_ROWS = 30  # 1 s


def _lengths(ranges):
  return [b - a + 1 for a, b in ranges]


def test_short_sequence_is_kept_whole():
  """"...while retaining shorter sequences as individual clips." """
  assert _slice_ranges(180, ROWS_PER_CLIP, MIN_ROWS) == [(1, 180)]
  # Even below the minimum: a whole source sequence is never dropped.
  assert _slice_ranges(10, ROWS_PER_CLIP, MIN_ROWS) == [(1, 10)]


def test_exact_multiple_splits_evenly():
  ranges = _slice_ranges(900, ROWS_PER_CLIP, MIN_ROWS)
  assert ranges == [(1, 300), (301, 600), (601, 900)]


def test_no_clip_ever_exceeds_the_limit():
  """Regression: folding a short tail into the previous slice produced 12.5 s clips.

  Anything longer than ``episode_length_s`` can never be tracked to completion, so
  stratification would score every such clip as a failure regardless of the policy.
  """
  for num_rows in range(ROWS_PER_CLIP + 1, ROWS_PER_CLIP * 4, 7):
    ranges = _slice_ranges(num_rows, ROWS_PER_CLIP, MIN_ROWS)
    assert max(_lengths(ranges)) <= ROWS_PER_CLIP, (num_rows, _lengths(ranges))


def test_short_tail_is_dropped_not_merged():
  # 300 + 10 rows: the 10-row tail is below MIN_ROWS.
  ranges = _slice_ranges(310, ROWS_PER_CLIP, MIN_ROWS)
  assert ranges == [(1, 300)]


def test_tail_at_or_above_minimum_is_kept():
  ranges = _slice_ranges(300 + MIN_ROWS, ROWS_PER_CLIP, MIN_ROWS)
  assert ranges == [(1, 300), (301, 330)]


def test_documented_command_lines_actually_parse():
  """The invocations printed in the docstring / CLAUDE.md must work verbatim.

  mjlab's tyro configuration disables implicit boolean flags, so a bare ``--append``
  fails with "Missing value for argument". That is exactly the command a user copies
  when ingesting a second motion source, and the failure is at argument-parse time
  with no hint that the docs are wrong.
  """
  cfg = tyro.cli(
    Config,
    args=[
      "--input-dir", "data/raw/seed",
      "--source", "seed",
      "--input-fps", "30",
      "--append", "True",
    ],
    config=mjlab.TYRO_FLAGS,
  )
  assert cfg.append is True
  assert cfg.source == "seed"
  assert cfg.input_fps == 30.0

  # The first-pass invocation, without --append.
  cfg = tyro.cli(
    Config,
    args=["--input-dir", "data/raw/lafan1", "--source", "lafan1", "--input-fps", "30"],
    config=mjlab.TYRO_FLAGS,
  )
  assert cfg.append is False
  assert cfg.clip_seconds == 10.0


def test_ranges_are_contiguous_and_non_overlapping():
  ranges = _slice_ranges(1000, ROWS_PER_CLIP, MIN_ROWS)
  # strict=False on purpose: pairing a list with its own tail is inherently ragged.
  for (_, prev_stop), (start, _) in zip(ranges, ranges[1:], strict=False):
    assert start == prev_stop + 1
  assert ranges[0][0] == 1
