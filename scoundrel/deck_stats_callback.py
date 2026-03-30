"""Log draw-pile and in-play monster stats at episode end (``rollout/deck_*``)."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _merged_info(info: object) -> dict:
    """VecEnv / Gymnasium may nest terminal fields under ``final_info``."""
    if not isinstance(info, dict):
        return {}
    fin = info.get("final_info")
    if isinstance(fin, dict):
        out = dict(info)
        out.update(fin)
        return out
    return info


class DeckStatsTensorboardCallback(BaseCallback):
    """At each rollout end, log mean deck rank-sum and mean monsters left in deck+hand."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._damage: list[float] = []
        self._remaining: list[float] = []

    def _on_step(self) -> bool:
        # VecEnv passes ``dones`` as a numpy array; ``arr or []`` is invalid.
        raw_dones = self.locals.get("dones")
        if raw_dones is None:
            dones: list | np.ndarray = []
        else:
            dones = np.asarray(raw_dones).astype(bool).reshape(-1)
        raw_infos = self.locals.get("infos")
        infos = [] if raw_infos is None else raw_infos
        for i, done in enumerate(dones):
            if i >= len(infos):
                break
            info = _merged_info(infos[i])
            if not done:
                continue
            if "deck_monster_damage_left" in info:
                self._damage.append(float(info["deck_monster_damage_left"]))
            if "monsters_remaining_in_play" in info:
                self._remaining.append(float(info["monsters_remaining_in_play"]))
        return True

    def _on_rollout_end(self) -> bool:
        if self._damage:
            self.logger.record(
                "rollout/deck_monster_damage_left_mean",
                float(np.mean(self._damage)),
            )
        if self._remaining:
            self.logger.record(
                "rollout/monsters_remaining_in_play_mean",
                float(np.mean(self._remaining)),
            )
        self._damage.clear()
        self._remaining.clear()
        return True
