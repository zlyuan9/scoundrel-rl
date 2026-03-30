import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scoundrel.card import Card
from scoundrel.engine import Engine

# Full deck size from ``create_deck()`` (must match engine).
INITIAL_DECK_SIZE = 44

# Potential-based shaping (Ng et al., ICML 1999): r' = r + γ Φ(s') − Φ(s).
# Use the same γ as the RL algorithm (e.g. PPO ``gamma``).
DEFAULT_POTENTIAL_SHAPING_GAMMA = 0.99

_SUIT_TO_IDX = {"spades": 0, "clubs": 1, "diamonds": 2, "hearts": 3}
_TYPE_TO_IDX = {"monster": 0, "weapon": 1, "potion": 2}
_IDX_TO_SUIT = ("spades", "clubs", "diamonds", "hearts")
_CHAIN_SENTINEL = 9999  # encodes +inf chain cap in weapon vector


def encode_hand(engine: Engine) -> np.ndarray:
    """Four slots × (rank, suit_idx, type_idx); empty slot = (-1,-1,-1)."""
    out = np.full((4, 3), -1, dtype=np.int32)
    for i in range(4):
        c = engine.hand[i]
        if c is None:
            continue
        out[i] = (c.value, _SUIT_TO_IDX[c.suit], _TYPE_TO_IDX[c.type])
    return out


def encode_weapon(engine: Engine) -> np.ndarray:
    """[rank, suit_idx, last_chain]; bare hands = (-1,-1,-1)."""
    w, last = engine.weapon
    if w is None:
        return np.array([-1, -1, -1], dtype=np.int32)
    last_enc = _CHAIN_SENTINEL if math.isinf(last) else int(last)
    return np.array([w.value, _SUIT_TO_IDX[w.suit], last_enc], dtype=np.int32)


def _rank_short(value: int) -> str:
    if value <= 10:
        return str(value)
    return {11: "J", 12: "Q", 13: "K", 14: "A"}[value]


def _suit_letter(suit: str) -> str:
    return {"spades": "S", "clubs": "C", "diamonds": "D", "hearts": "H"}.get(suit, "?")

def _type_tag(card: Card) -> str:
    if card.type == "monster":
        return "FOE"
    if card.type == "weapon":
        return "WEA"
    return "POT"


def _format_weapon(weapon_tuple) -> str:
    wcard, last = weapon_tuple
    if wcard is None:
        return "bare hands"
    chain = "inf" if math.isinf(last) else str(int(last))
    return f"{_rank_short(wcard.value)}{_suit_letter(wcard.suit)} (chain<{chain})"


# Fixed width between "|" and "|" so ASCII box edges line up in monospaced terminals.
_BOX_INNER = 42


def _box_top() -> str:
    return "+" + "-" * _BOX_INNER + "+"


def _box_row(inner: str) -> str:
    inner = inner.replace("\n", " ")[:_BOX_INNER].ljust(_BOX_INNER)
    return "|" + inner + "|"


def _format_card_slot(i: int, card: Card | None) -> str:
    if card is None:
        return f"  [{i}]     (empty)"
    body = f"{_rank_short(card.value)}{_suit_letter(card.suit)}"
    return f"  [{i}]   {body:>4}   {_type_tag(card):>3}"


def _weapon_tuple_from_array(w: np.ndarray):
    """Rebuild (Card|None, last) for printing from encoded weapon vector."""
    w = np.asarray(w, dtype=np.int64).ravel()
    if w[0] < 0:
        return (None, math.inf)
    suit = _IDX_TO_SUIT[int(w[1])]
    card = Card(suit, int(w[0]), "x")
    last = math.inf if w[2] >= _CHAIN_SENTINEL else float(w[2])
    return (card, last)


def _card_from_hand_row(row: np.ndarray) -> Card | None:
    row = np.asarray(row, dtype=np.int64).ravel()
    if row[0] < 0:
        return None
    suit = _IDX_TO_SUIT[int(row[1])]
    return Card(suit, int(row[0]), "x")


def format_observation(obs: dict, *, engine: Engine | None = None) -> str:
    """Pretty-print a :class:`ScoundrelEnv` observation dict (ASCII, terminal-friendly).

    Pass ``engine`` for extra lines (resolves left, phase) and simplest card display.
    Without ``engine``, weapon/hand are decoded from numeric ``obs`` when possible.
    """
    mask = np.asarray(obs["action_mask"], dtype=bool)
    deck_n = int(np.asarray(obs["remaining_cards"]).ravel()[0])
    hp = int(np.asarray(obs["health"]).ravel()[0])
    skipped = bool(np.asarray(obs["skipped_room"]).ravel()[0])
    if engine is not None:
        weapon_t = engine.weapon
        hand = engine.hand
    else:
        weapon_t = _weapon_tuple_from_array(obs["weapon"])
        hand = {
            i: _card_from_hand_row(obs["hand"][i])
            for i in range(4)
        }

    lines: list[str] = []
    lines.append(_box_top())
    lines.append(_box_row("  SCOUNDREL (obs)"))
    lines.append(_box_top())
    lines.append(_box_row(f"  HP {hp:>2}/20     dungeon cards: {deck_n:>2}"))
    lines.append(_box_row(f"  weapon: {_format_weapon(weapon_t)}"))
    lines.append(_box_row(f"  skipped last room: {skipped}"))
    if engine is not None:
        phase = (
            "room (resolve)"
            if engine._is_room_phase()
            else ("turn (skip/play)" if not (engine.game_won or engine.game_lost) else "over")
        )
        lines.append(_box_row(f"  phase: {phase}"))
        lines.append(_box_row(f"  resolves left: {engine._resolves_left}"))
    lines.append(_box_top())
    lines.append(_box_row("  ROOM"))
    for i in range(4):
        c = hand.get(i)
        lines.append(_box_row(_format_card_slot(i, c)))
    lines.append(_box_top())
    labels = "sk ent s0 s1 s2 s3"
    dots = " ".join("@" if mask[j] else "." for j in range(6))
    lines.append(_box_row(f"  legal   {labels}"))
    lines.append(_box_row(f"          {dots}"))
    lines.append(_box_top())
    return "\n".join(lines)


class ScoundrelEnv(gym.Env):
    """Gymnasium env for Scoundrel.

    Optional **potential-based shaping** uses Φ(s) = (D − |deck|) / D with D = ``INITIAL_DECK_SIZE``,
    so progress through the draw pile increases potential. Shaping is skipped when the engine rejects
    an action (invalid mask) so Φ(s') = Φ(s) does not inject (γ−1)Φ noise.
    """

    metadata = {
        "render_modes": ("human", "ansi", "rgb_array"),
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode=None,
        *,
        potential_shaping: bool = True,
        potential_shaping_gamma: float = DEFAULT_POTENTIAL_SHAPING_GAMMA,
    ):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}; got {render_mode!r}"
            )
        self.render_mode = render_mode
        self.potential_shaping = potential_shaping
        self.potential_shaping_gamma = potential_shaping_gamma
        self._prev_phi = 0.0
        self.engine = Engine()
        self._viewer = None
        self._offscreen_surface = None
        self._offscreen_view = None
        self.action_mask = None
        self.observation_space = spaces.Dict(
            {
                "remaining_cards": spaces.Box(low=0, high=44, shape=(1,), dtype=np.int32),
                "hand": spaces.Box(low=-1, high=14, shape=(4, 3), dtype=np.int32),
                "health": spaces.Box(low=0, high=20, shape=(1,), dtype=np.int32),
                "weapon": spaces.Box(low=-1, high=10000, shape=(3,), dtype=np.int32),
                "skipped_room": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32),
                "action_mask": spaces.MultiBinary(6),
            }
        )

        '''
        Action space:
          0: skip room
          1: enter room (begin_play_room)
          2–5: resolve hand slots 0–3 (only valid in room phase)
        '''
        self.action_space = spaces.Discrete(6)

    def _deck_potential(self) -> float:
        """Φ(s) in [0, 1]: higher when fewer cards remain in the draw pile."""
        d = len(self.engine.deck)
        return float(INITIAL_DECK_SIZE - d) / float(INITIAL_DECK_SIZE)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        self.action_mask = np.array([True, True, False, False, False, False])
        self._prev_phi = self._deck_potential()
        return self._get_obs(), {}

    def step(self, action):
        phi_s = self._prev_phi
        reward, terminated, truncated, info = self.engine.step(action)
        phi_sp = self._deck_potential()
        if not info.get("invalid_action"):
            if self.potential_shaping:
                reward += self.potential_shaping_gamma * phi_sp - phi_s
            self._prev_phi = phi_sp
        if self.engine._is_turn_phase():
            action_mask = np.array([True, True, False, False, False, False])
        elif self.engine._is_room_phase():
            action_mask = np.array([False, False, self.engine.hand[0] is not None, self.engine.hand[1] is not None, self.engine.hand[2] is not None, self.engine.hand[3] is not None])
        else:
            action_mask = np.array([False, False, False, False, False, False])
        self.action_mask = action_mask
        return self._get_obs(), reward, terminated, truncated, info

    def _render_text(self) -> str:
        """Plain-text snapshot for ``ansi`` mode (no pygame)."""
        e = self.engine
        wcard, last_slain = e.weapon
        if wcard is None:
            weapon_line = "Weapon: none"
        else:
            weapon_line = f"Weapon: {wcard} (last kill: {last_slain})"
        lines = [
            "=== Scoundrel ===",
            f"Health: {e.health}  |  Deck: {len(e.deck)} cards",
            weapon_line,
            f"Skipped room (last turn): {e.skipped_room}  |  Potion used this room: {e.used_potion}",
            f"Resolves left in room: {e._resolves_left}",
            "--- Room ---",
        ]
        for i in range(4):
            c = e.hand[i]
            slot = "empty" if c is None else str(c)
            lines.append(f"  [{i}] {slot}")
        lines.append("--- Status ---")
        if e.game_won:
            lines.append("Status: WON")
        elif e.game_lost:
            lines.append("Status: LOST")
        else:
            lines.append("Status: playing")
        return "\n".join(lines)

    def render(self):
        """Render via the pygame viewer (``viewer.main.GameView``).

        - ``human``: windowed viewer; returns ``None``.
        - ``ansi``: plain text only (no pygame); for logging / headless tests.
        - ``rgb_array``: same layout as the viewer, ``(H, W, 3)`` ``uint8`` offscreen.

        Requires ``pip install 'scoundrel-rl[gui]'`` for ``human`` and ``rgb_array``.
        """
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_text()

        try:
            import pygame
            from viewer.main import GameView, W, H
        except ImportError as err:
            raise ImportError(
                "Viewer render requires pygame. Install with: pip install 'scoundrel-rl[gui]'"
            ) from err

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = GameView()
            self._viewer.render_engine(self.engine, message="RL", log=[])
            pygame.display.flip()
            pygame.event.pump()
            return None

        if self.render_mode == "rgb_array":
            if self._offscreen_surface is None:
                self._offscreen_surface = pygame.Surface((W, H))
            if self._offscreen_view is None:
                self._offscreen_view = GameView(surface=self._offscreen_surface)
            self._offscreen_view.render_engine(self.engine, message="RL", log=[])
            arr = pygame.surfarray.array3d(self._offscreen_surface)
            return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

        raise RuntimeError(f"Unhandled render_mode: {self.render_mode!r}")

    def close(self):
        """Release pygame if the viewer was used."""
        self._viewer = None
        self._offscreen_view = None
        self._offscreen_surface = None
        try:
            import pygame

            if pygame.get_init():
                pygame.quit()
        except Exception:
            pass

    def _get_obs(self):
        return {
            "remaining_cards": np.array([len(self.engine.deck)], dtype=np.int32),
            "hand": encode_hand(self.engine),
            "health": np.array([self.engine.health], dtype=np.int32),
            "weapon": encode_weapon(self.engine),
            "skipped_room": np.array([int(self.engine.skipped_room)], dtype=np.int32),
            "action_mask": np.asarray(self.action_mask, dtype=np.int8),
        }

   