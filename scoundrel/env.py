import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scoundrel.card import Card
from scoundrel.engine import Engine

# Full deck size from ``create_deck()`` (must match engine).
INITIAL_DECK_SIZE = 44
# Monsters only: spades + clubs, ranks 2–14 → 13×2 (see ``create_deck``).
TOTAL_MONSTER_CARDS = 26

DEFAULT_WIN_REWARD = 1.0
DEFAULT_LOSS_REWARD = -2.0
DEFAULT_DECK_DAMAGE_LEAVE_REWARD = 0.003
DEFAULT_ROOM_HP_LOSS_PENALTY_PER_HP = 0.03
DEFAULT_POTENTIAL_SHAPING_GAMMA = 0.99
DEFAULT_POTENTIAL_SHAPING_SCALE = 0.15
DEFAULT_WASTED_POTION_PENALTY = 0.05
DEFAULT_POTION_OVERHEAL_PENALTY_PER_HP = 0.02
DEFAULT_WEAPON_DOWNGRADE_PENALTY = 0.05
DEFAULT_ROOM_SURVIVED_BONUS = 0.0

# ---------------------------------------------------------------------------
# Flat observation encoding — 33 normalised float32 features
# ---------------------------------------------------------------------------
# Per card slot (×4 = 20): occupied, rank/14, is_monster, is_weapon, is_potion
# Weapon (3): has_weapon, value/14, chain_max/14 (inf → 1.0)
# Global (5): health/20, remaining/44, skipped_room, used_potion, resolves_left/3
# Deck composition (5): monsters_in_deck/26, avg_monster_rank (0-1),
#                        weapons_in_deck/9, max_weapon_rank (0-1), potions_in_deck/9
OBS_DIM = 33
_SLOT_FEAT = 5  # features per card slot
_WEAPON_OFF = 20
_GLOBAL_OFF = 23
_DECK_COMP_OFF = 28
_MAX_RANK = 14.0


def encode_obs_flat(engine: Engine) -> np.ndarray:
    """Encode full game state as a normalised [0, 1] float32 vector."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    for i in range(4):
        card = engine.hand[i]
        base = i * _SLOT_FEAT
        if card is not None:
            obs[base] = 1.0
            obs[base + 1] = card.value / _MAX_RANK
            obs[base + 2] = float(card.type == "monster")
            obs[base + 3] = float(card.type == "weapon")
            obs[base + 4] = float(card.type == "potion")
    wcard, last_slain = engine.weapon
    if wcard is not None:
        obs[_WEAPON_OFF] = 1.0
        obs[_WEAPON_OFF + 1] = wcard.value / _MAX_RANK
        obs[_WEAPON_OFF + 2] = 1.0 if math.isinf(last_slain) else last_slain / _MAX_RANK
    obs[_GLOBAL_OFF] = engine.health / 20.0
    obs[_GLOBAL_OFF + 1] = len(engine.deck) / float(INITIAL_DECK_SIZE)
    obs[_GLOBAL_OFF + 2] = float(engine.skipped_room)
    obs[_GLOBAL_OFF + 3] = float(engine.used_potion)
    obs[_GLOBAL_OFF + 4] = engine._resolves_left / 3.0
    # Deck composition — what types of cards remain (card-counting info).
    n_monsters = 0
    monster_rank_sum = 0
    n_weapons = 0
    max_weapon_rank = 0
    n_potions = 0
    for c in engine.deck:
        if c.type == "monster":
            n_monsters += 1
            monster_rank_sum += c.value
        elif c.type == "weapon":
            n_weapons += 1
            max_weapon_rank = max(max_weapon_rank, c.value)
        else:
            n_potions += 1
    obs[_DECK_COMP_OFF] = n_monsters / 26.0
    obs[_DECK_COMP_OFF + 1] = (monster_rank_sum / max(n_monsters, 1)) / _MAX_RANK
    obs[_DECK_COMP_OFF + 2] = n_weapons / 9.0
    obs[_DECK_COMP_OFF + 3] = max_weapon_rank / _MAX_RANK
    obs[_DECK_COMP_OFF + 4] = n_potions / 9.0
    return obs


# ---------------------------------------------------------------------------
# Legacy helpers kept for render / format / tests that import them
# ---------------------------------------------------------------------------
_SUIT_TO_IDX = {"spades": 0, "clubs": 1, "diamonds": 2, "hearts": 3}
_TYPE_TO_IDX = {"monster": 0, "weapon": 1, "potion": 2}
_IDX_TO_SUIT = ("spades", "clubs", "diamonds", "hearts")
_CHAIN_SENTINEL = 9999
# VecEnv autoreset calls ``reset()`` with ``seed=None``; sample deck RNG seeds from [0, 2**63)
# so episodes are not tied to a small modulus (SB3 still passes explicit seeds on first reset).
_EPISODE_SEED_UPPER = 2**63


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


def format_observation(obs, *, engine: Engine | None = None) -> str:
    """Pretty-print observation. Works with both flat ndarray and legacy dict obs.

    Always pass ``engine`` for complete output (required for flat obs format).
    """
    if engine is not None:
        weapon_t = engine.weapon
        hand = engine.hand
        hp = engine.health
        deck_n = len(engine.deck)
        skipped = engine.skipped_room
        mask = None
    elif isinstance(obs, dict):
        weapon_t = _weapon_tuple_from_array(obs["weapon"])
        hand = {i: _card_from_hand_row(obs["hand"][i]) for i in range(4)}
        hp = int(np.asarray(obs["health"]).ravel()[0])
        deck_n = int(np.asarray(obs["remaining_cards"]).ravel()[0])
        skipped = bool(np.asarray(obs["skipped_room"]).ravel()[0])
        mask = np.asarray(obs.get("action_mask", np.zeros(6)), dtype=bool)
    else:
        # Flat obs — need engine for full decode
        flat = np.asarray(obs).ravel()
        hp = int(round(flat[_GLOBAL_OFF] * 20))
        deck_n = int(round(flat[_GLOBAL_OFF + 1] * INITIAL_DECK_SIZE))
        skipped = flat[_GLOBAL_OFF + 2] > 0.5
        weapon_t = (None, math.inf)
        hand = {i: None for i in range(4)}
        mask = None

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
        c = hand.get(i) if isinstance(hand, dict) else hand[i]
        lines.append(_box_row(_format_card_slot(i, c)))
    lines.append(_box_top())
    if mask is not None:
        labels = "sk ent s0 s1 s2 s3"
        dots = " ".join("@" if mask[j] else "." for j in range(6))
        lines.append(_box_row(f"  legal   {labels}"))
        lines.append(_box_row(f"          {dots}"))
        lines.append(_box_top())
    return "\n".join(lines)


def _weapon_tuple_from_array(w: np.ndarray):
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


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _monster_damage_sum_in_deck(engine: Engine) -> int:
    """Sum of monster card ranks (damage) still in the draw pile."""
    return sum(c.value for c in engine.deck if c.type == "monster")


def _count_monsters_in_deck_only(engine: Engine) -> int:
    """Monsters still in the draw pile (not hand / discard)."""
    return sum(1 for c in engine.deck if c.type == "monster")


def _count_monsters_in_deck_and_hand(engine: Engine) -> int:
    """Monsters still in play: draw pile plus room hand (not yet discarded)."""
    n = 0
    for c in engine.deck:
        if c.type == "monster":
            n += 1
    for i in range(4):
        c = engine.hand[i]
        if c is not None and c.type == "monster":
            n += 1
    return n


def _monster_progress_potential(engine: Engine) -> float:
    """Φ(s) = 1 − rem/26 with rem = monsters in deck + hand."""
    rem = _count_monsters_in_deck_and_hand(engine)
    return 1.0 - (float(rem) / float(TOTAL_MONSTER_CARDS))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ScoundrelEnv(gym.Env):
    """Gymnasium env for Scoundrel with normalised flat observation.

    Observation is a float32 vector of length 28 (see ``encode_obs_flat``).
    Action mask is provided via ``self.action_mask`` (read by ``ActionMasker``).
    """

    metadata = {
        "render_modes": ("human", "ansi", "rgb_array"),
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode=None,
        *,
        win_reward: float = DEFAULT_WIN_REWARD,
        loss_reward: float = DEFAULT_LOSS_REWARD,
        deck_damage_leave_reward: float = DEFAULT_DECK_DAMAGE_LEAVE_REWARD,
        room_hp_loss_penalty_per_hp: float = DEFAULT_ROOM_HP_LOSS_PENALTY_PER_HP,
        sparse_reward_only: bool = False,
        potential_shaping: bool = True,
        potential_shaping_gamma: float = DEFAULT_POTENTIAL_SHAPING_GAMMA,
        potential_shaping_scale: float = DEFAULT_POTENTIAL_SHAPING_SCALE,
        wasted_potion_penalty: float = DEFAULT_WASTED_POTION_PENALTY,
        potion_overheal_penalty_per_hp: float = DEFAULT_POTION_OVERHEAL_PENALTY_PER_HP,
        weapon_downgrade_penalty: float = DEFAULT_WEAPON_DOWNGRADE_PENALTY,
        room_survived_bonus: float = DEFAULT_ROOM_SURVIVED_BONUS,
    ):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']}; got {render_mode!r}"
            )
        self.render_mode = render_mode
        self.win_reward = float(win_reward)
        self.loss_reward = float(loss_reward)
        self.deck_damage_leave_reward = float(deck_damage_leave_reward)
        self.room_hp_loss_penalty_per_hp = float(room_hp_loss_penalty_per_hp)
        self.sparse_reward_only = bool(sparse_reward_only)
        self.potential_shaping = bool(potential_shaping)
        self.potential_shaping_gamma = float(potential_shaping_gamma)
        self.potential_shaping_scale = float(potential_shaping_scale)
        self.wasted_potion_penalty = float(wasted_potion_penalty)
        self.potion_overheal_penalty_per_hp = float(potion_overheal_penalty_per_hp)
        self.weapon_downgrade_penalty = float(weapon_downgrade_penalty)
        self.room_survived_bonus = float(room_survived_bonus)
        self._prev_deck_monster_damage = 0
        self._health_at_room_start: float | None = None
        self._prev_phi = 0.0
        self.engine = Engine(win_reward=self.win_reward, loss_reward=self.loss_reward)
        self._viewer = None
        self._offscreen_surface = None
        self._offscreen_view = None
        self.action_mask = None

        # Flat normalised observation — no action_mask in obs (provided via wrapper).
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        # 0: skip, 1: enter room, 2–5: resolve slots 0–3
        self.action_space = spaces.Discrete(6)

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = int(
                self.np_random.integers(0, _EPISODE_SEED_UPPER, dtype=np.uint64)
            )
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        # After reset we're in turn phase; skip is legal (skipped_room=False).
        self.action_mask = np.array([True, True, False, False, False, False])
        self._prev_deck_monster_damage = _monster_damage_sum_in_deck(self.engine)
        self._health_at_room_start = None
        self._prev_phi = _monster_progress_potential(self.engine)
        return self._get_obs(), {}

    def step(self, action):
        e = self.engine
        resolves_before = e._resolves_left
        in_room_before = resolves_before > 0

        reward, terminated, truncated, info = e.step(action)
        info = dict(info)
        curr_dmg = _monster_damage_sum_in_deck(e)
        info["deck_monster_damage_left"] = float(curr_dmg)
        if not info.get("invalid_action"):
            if not self.sparse_reward_only:
                delta = self._prev_deck_monster_damage - curr_dmg
                if delta > 0:
                    reward += self.deck_damage_leave_reward * float(delta)

                if resolves_before == 0 and e._resolves_left == 3:
                    self._health_at_room_start = float(e.health)

                room_ended = in_room_before and (
                    e.game_lost or e.game_won or e._resolves_left == 0
                )
                if room_ended and self._health_at_room_start is not None:
                    end_hp = max(0.0, float(e.health))
                    hp_lost = max(0.0, self._health_at_room_start - end_hp)
                    reward -= self.room_hp_loss_penalty_per_hp * hp_lost
                    # Bonus for surviving the room
                    if not e.game_lost:
                        reward += self.room_survived_bonus
                    self._health_at_room_start = None

                if self.wasted_potion_penalty > 0.0 and info.get("wasted_potion"):
                    reward -= self.wasted_potion_penalty
                oh = info.get("potion_overheal")
                if (
                    self.potion_overheal_penalty_per_hp > 0.0
                    and isinstance(oh, (int, float))
                    and oh > 0
                ):
                    reward -= self.potion_overheal_penalty_per_hp * float(oh)
                if self.weapon_downgrade_penalty > 0.0 and info.get("weapon_downgrade"):
                    reward -= self.weapon_downgrade_penalty

                if self.potential_shaping:
                    phi_s = self._prev_phi
                    phi_sp = _monster_progress_potential(e)
                    reward += self.potential_shaping_scale * (
                        self.potential_shaping_gamma * phi_sp - phi_s
                    )
                    self._prev_phi = phi_sp
            self._prev_deck_monster_damage = curr_dmg

        if terminated or truncated:
            info["episode_win"] = bool(self.engine.game_won)
            info["monsters_remaining_in_play"] = float(
                _count_monsters_in_deck_and_hand(self.engine)
            )

        # --- Action mask (skip is only legal when we haven't just skipped) ---
        if self.engine._is_turn_phase():
            can_skip = not self.engine.skipped_room
            action_mask = np.array([can_skip, True, False, False, False, False])
        elif self.engine._is_room_phase():
            action_mask = np.array(
                [
                    False,
                    False,
                    self.engine.hand[0] is not None,
                    self.engine.hand[1] is not None,
                    self.engine.hand[2] is not None,
                    self.engine.hand[3] is not None,
                ]
            )
        else:
            action_mask = np.array([False, False, False, False, False, False])
        self.action_mask = action_mask
        return self._get_obs(), reward, terminated, truncated, info

    def _render_text(self) -> str:
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
        return encode_obs_flat(self.engine)
