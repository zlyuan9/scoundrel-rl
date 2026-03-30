"""Log rolling mean training win rate to TensorBoard (``rollout/win_rate_mean``)."""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class WinRateTensorboardCallback(BaseCallback):
    """Append ``episode_win`` from env info and log mean win rate each rollout end."""

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self._recent: deque[float] = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict) or "episode_win" not in info:
                continue
            self._recent.append(1.0 if info["episode_win"] else 0.0)
        return True

    def _on_rollout_end(self) -> bool:
        if self._recent:
            self.logger.record(
                "rollout/win_rate_mean",
                float(np.mean(self._recent)),
            )
        return True
