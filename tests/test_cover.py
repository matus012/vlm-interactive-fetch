import math

import pytest

from sim.cover_check import Cover
from sim.world.generate import load_places, room_point


@pytest.fixture(scope="module")
def cov() -> Cover:
    return Cover()


def test_box_in_front_counted(cov: Cover) -> None:
    """Living-room point, heading 0 faces +x: a box 1.5 m ahead is >= 100 px in the heading-0 photo."""
    x, y = room_point("living_room", load_places())
    px = cov.px((x, y), [(x + 1.5, y, 0.1, 0.0)], True)[0]
    assert px[0] >= 100, px


def test_box_behind_wall_not_counted(cov: Cover) -> None:
    """Same stand, box in the bedroom (other side of walls): 0 px in all four photos."""
    x, y = room_point("living_room", load_places())
    bx, by = room_point("bedroom", load_places())
    px = cov.px((x, y), [(bx, by, 0.1, 0.0)], True)[0]
    assert px.max() == 0, px
