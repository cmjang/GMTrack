from __future__ import annotations

import pytest

from ex_grmt.scripts.webplay import _manual_push_velocity_range


@pytest.mark.parametrize(
  ("direction", "expected"),
  [
    ("+X", {"x": (1.25, 1.25)}),
    ("-X", {"x": (-1.25, -1.25)}),
    ("+Y", {"y": (1.25, 1.25)}),
    ("-Y", {"y": (-1.25, -1.25)}),
  ],
)
def test_manual_push_velocity_range(direction, expected):
  assert _manual_push_velocity_range(direction, 1.25) == expected


def test_manual_push_rejects_invalid_input():
  with pytest.raises(ValueError, match="positive"):
    _manual_push_velocity_range("+X", 0.0)
  with pytest.raises(ValueError, match="direction"):
    _manual_push_velocity_range("up", 1.0)
