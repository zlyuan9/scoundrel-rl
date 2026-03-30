"""Shared fixtures for engine tests (no full deck / no stdin)."""

from __future__ import annotations

import math

import pytest

from scoundrel.engine import Engine


@pytest.fixture
def make_engine():
    """Build a minimal Engine instance without __init__ (no shuffled deck)."""

    def _make(**overrides):
        e = Engine.__new__(Engine)
        e.deck = []
        e.health = 20
        e.game_won = False
        e.game_lost = False
        e.weapon = (None, math.inf)
        e.hand = {0: None, 1: None, 2: None, 3: None}
        e.skipped_room = False
        e.used_potion = False
        e._resolves_left = 0
        e._last_resolve_info = {}
        for k, v in overrides.items():
            setattr(e, k, v)
        return e

    return _make
