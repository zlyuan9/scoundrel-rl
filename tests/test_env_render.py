import numpy as np
import pytest

from scoundrel.env import ScoundrelEnv


def test_render_ansi_returns_nonempty_string():
    env = ScoundrelEnv(render_mode="ansi")
    s = env.render()
    assert isinstance(s, str)
    assert len(s) > 0
    assert "Scoundrel" in s


def test_render_human_returns_none(monkeypatch):
    """Avoid opening a real window; ensure viewer path runs."""
    calls = []

    class FakeView:
        def __init__(self, *, surface=None):
            self._owns_window = surface is None

        def render_engine(self, engine, **kwargs):
            calls.append(engine)

    monkeypatch.setattr("viewer.main.GameView", FakeView)
    monkeypatch.setattr("pygame.display.flip", lambda: None)
    monkeypatch.setattr("pygame.event.pump", lambda: None)

    env = ScoundrelEnv(render_mode="human")
    assert env.render() is None
    assert len(calls) == 1
    assert calls[0] is env.engine


def test_render_none_mode_returns_none():
    env = ScoundrelEnv(render_mode=None)
    assert env.render() is None


def test_render_rgb_array_shape_and_dtype():
    try:
        import pygame  # noqa: F401
    except ImportError:
        pytest.skip("pygame not installed")
    env = ScoundrelEnv(render_mode="rgb_array")
    arr = env.render()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8
    assert arr.shape[0] == 640 and arr.shape[1] == 960
