from scoundrel.card import Card
import random
from random import shuffle
import math


def create_deck():
    deck = []
    for i in range(2, 15):
        deck.append(Card("spades", i, "monster"))
        deck.append(Card("clubs", i, "monster"))
    for i in range(2, 11):
        deck.append(Card("diamonds", i, "weapon"))
    for i in range(2, 11):
        deck.append(Card("hearts", i, "potion"))
    return deck


class Engine:
    def __init__(
        self,
        *,
        seed: int | None = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
    ):
        self._win_reward = float(win_reward)
        self._loss_reward = float(loss_reward)
        if seed is not None:
            random.seed(seed)
        self._init_state()

    def _init_state(self) -> None:
        self.deck = create_deck()
        shuffle(self.deck)
        self.health = 20
        self.game_won = False
        self.game_lost = False
        self.weapon = (None, math.inf)  # (weapon card, max monster value killed with weapon)
        self.hand = {0: None, 1: None, 2: None, 3: None}
        self.skipped_room = False
        self.used_potion = False
        self._resolves_left = 0
        self._last_resolve_info: dict = {}

    def __str__(self):
        return f"Engine(deck={self.deck})"

    def __repr__(self):
        return f"Engine(deck={self.deck})"

    def get_deck(self):
        return self.deck

    def reset(self, seed: int | None = None) -> None:
        """Reset the dungeon and deal the first room (same as a new RL episode)."""
        if seed is not None:
            random.seed(seed)
        self._init_state()
        self.start_turn()

    def _is_turn_phase(self) -> bool:
        return (
            not self.game_won
            and not self.game_lost
            and self._resolves_left == 0
        )

    def _is_room_phase(self) -> bool:
        return self._resolves_left > 0

    def _reward_for_terminal(self) -> float:
        if self.game_won:
            return self._win_reward
        if self.game_lost:
            return self._loss_reward
        return 0.0

    def step(self, action: int) -> tuple[float, bool, bool, dict]:
        """Apply one RL action. Actions: 0 skip, 1 enter room, 2–5 resolve slots 0–3.

        Returns ``(reward, terminated, truncated, info)``. Matches viewer semantics when
        skip is illegal (forced into the room).
        """
        if self.game_won or self.game_lost:
            return 0.0, True, False, {}

        if self._is_room_phase():
            if action not in (2, 3, 4, 5):
                return 0.0, False, False, {"invalid_action": True}
            slot = action - 2
            if self.hand[slot] is None:
                return 0.0, False, False, {"invalid_action": True}
            result = self.resolve_in_room(slot)
            if result == "room_done":
                self.start_turn()
            reward = self._reward_for_terminal()
            terminated = self.game_won or self.game_lost
            return reward, terminated, False, dict(self._last_resolve_info)

        # Turn phase: choose skip or enter room
        if action not in (0, 1):
            return 0.0, False, False, {"invalid_action": True}

        if action == 0:
            if not self.try_skip():
                self.begin_play_room()
            else:
                self.start_turn()
        else:
            self.begin_play_room()

        reward = self._reward_for_terminal()
        terminated = self.game_won or self.game_lost
        return reward, terminated, False, {}

    def _fill_room(self) -> None:
        """Build a room of 4 cards: first room draws 4; later rooms keep the foundation + draw 3."""
        present = [(i, self.hand[i]) for i in range(4) if self.hand[i] is not None]
        n = len(present)

        if n == 4:
            return

        if n == 0:
            if len(self.deck) < 4:
                self.game_won = True
                return
            for i in range(4):
                self.hand[i] = self.deck.pop(0)
            return

        if n == 1:
            _, foundation = present[0]
            if len(self.deck) < 3:
                self.game_won = True
                return
            for i in range(4):
                self.hand[i] = None
            self.hand[0] = foundation
            for i in range(1, 4):
                self.hand[i] = self.deck.pop(0)
            return

        raise RuntimeError(f"invalid hand state: {n} cards in room (expected 0, 1, or 4)")

    def start_turn(self) -> bool:
        """Fill the room for this turn. Return True if the game is won (dungeon exhausted)."""
        self._fill_room()
        return self.game_won

    def try_skip(self) -> bool:
        """Move the current room to the bottom of the deck. Return False if skip is not allowed."""
        if self.skipped_room:
            return False
        cards = [self.hand[i] for i in range(4)]
        self.deck.extend(cards)
        self.hand = {0: None, 1: None, 2: None, 3: None}
        self.skipped_room = True
        return True

    def begin_play_room(self) -> None:
        """Start resolving cards in the current room (three picks)."""
        self.used_potion = False
        self._resolves_left = 3

    def resolve_in_room(self, card_index: int) -> str:
        """Resolve one card while playing a room. Returns playing | lost | room_done | won."""
        if self._resolves_left <= 0:
            raise RuntimeError("begin_play_room() was not called or room is already resolved")
        if self.hand[card_index] is None:
            raise RuntimeError("cannot resolve empty slot")
        self.resolve_card(card_index)
        self._resolves_left -= 1
        if self.game_lost:
            return "lost"
        if self._resolves_left == 0:
            self.skipped_room = False
            if len(self.deck) == 0:
                self.game_won = True
                return "won"
            return "room_done"
        return "playing"

    def can_use_weapon_on_monster(self, card: Card) -> bool:
        """True if an equipped weapon may be used on this monster (chain rule: m ≤ last slain).

        ``last_slain`` is ``math.inf`` after equipping until the first kill with that weapon.
        """
        if card.type != "monster":
            return False
        wcard, last_slain = self.weapon
        return wcard is not None and card.value <= last_slain

    def resolve_card(self, card_index):
        """Resolve one room card. Weapon chain: each kill must be ≤ the previous kill's rank with
        that weapon (DEVELOPER_GUIDE §2.6). New weapon resets the chain (last_slain = inf).

        Sets ``_last_resolve_info`` for potions: ``wasted_potion`` (second potion same room) or
        ``potion_overheal`` (HP that would exceed 20, as an int). For weapons: ``weapon_downgrade``
        when equipping a lower-rank weapon over an already-equipped higher-rank one.
        """
        self._last_resolve_info = {}
        card = self.hand[card_index]
        self.hand[card_index] = None
        if card.type == "monster":
            if self.can_use_weapon_on_monster(card):
                wcard, _last_slain = self.weapon
                dmg = max(0, card.value - wcard.value)
                self.health -= dmg
                self.weapon = (wcard, card.value)
                if self.health <= 0:
                    self.game_lost = True
                return
            self.health -= card.value
            if self.health <= 0:
                self.game_lost = True
            return
        if card.type == "weapon":
            wcard, _ = self.weapon
            if wcard is not None and card.value < wcard.value:
                self._last_resolve_info["weapon_downgrade"] = True
            self.weapon = (card, math.inf)
            return
        if card.type == "potion":
            if self.used_potion:
                self._last_resolve_info["wasted_potion"] = True
                return
            new_h = self.health + card.value
            overheal = max(0, new_h - 20)
            self.health = min(new_h, 20)
            self.used_potion = True
            if overheal > 0:
                self._last_resolve_info["potion_overheal"] = overheal
            return
        return


def run_terminal_cli() -> None:
    """stdin/stdout loop: get input, then call engine methods (no input inside Engine)."""
    e = Engine()
    if e.start_turn():
        print("Game won")
        return

    while not e.game_won and not e.game_lost:
        print("hand: ", e.hand)
        print("cards remaining: ", len(e.deck))
        print("health: ", e.health)
        print("weapon: ", e.weapon)

        if e._resolves_left == 0:
            choice = input("Choose to skip room or play room (s/p): ").strip().lower()
            if choice == "s":
                if e.skipped_room:
                    print("Cannot skip room twice in a row")
                    e.begin_play_room()
                else:
                    e.try_skip()
                    if e.start_turn():
                        print("Game won")
                        return
                continue
            if choice == "p":
                e.begin_play_room()
            else:
                print("Invalid input")
                continue
        else:
            raw = input("Choose to resolve card: ").strip()
            try:
                idx = int(raw)
            except ValueError:
                print("Invalid input")
                continue
            result = e.resolve_in_room(idx)
            if result == "lost":
                print("Game lost")
                break
            if result == "won":
                print("Game won")
                break
            if result == "room_done":
                e.start_turn()
                if e.game_won:
                    print("Game won")
                    break


if __name__ == "__main__":  # pragma: no cover
    random.seed(1)
    run_terminal_cli()
