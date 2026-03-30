"""
End-to-end integration tests: two scripted games (one loss, one win) with
assertions after each room and after key resolves.

These tests drive `resolve_card` and deck/hand state the same way as the RL
`Engine.step` / viewer loop, so we can validate rules before policy training.
"""

from __future__ import annotations

import math

from scoundrel.card import Card
from scoundrel.engine import Engine


def C(suit: str, value: int) -> Card:
    """Short Card constructor (third arg is ignored by Card; type comes from suit)."""
    return Card(suit, value, "x")


def fresh_hand() -> dict[int, Card | None]:
    return {0: None, 1: None, 2: None, 3: None}


def deal_room_from_deck(engine: Engine) -> None:
    """Fill hand slots 0–3 with the next four cards from engine.deck (front = pop(0))."""
    engine.hand = fresh_hand()
    for i in range(4):
        assert engine.deck, "deck exhausted before room could be filled"
        engine.hand[i] = engine.deck.pop(0)
    engine.used_potion = False


def deal_room_with_foundation(engine: Engine, foundation: Card) -> None:
    """Next room: one foundation card from the previous room + three draws from the deck."""
    engine.hand = fresh_hand()
    engine.hand[0] = foundation
    for i in (1, 2, 3):
        assert engine.deck, "deck exhausted before room could be filled"
        engine.hand[i] = engine.deck.pop(0)
    engine.used_potion = False


def play_three_resolves(engine: Engine, a: int, b: int, c: int) -> None:
    """Resolve three cards in order (indices refer to slots before each pop)."""
    for idx in (a, b, c):
        assert not engine.game_lost
        engine.resolve_card(idx)


def assert_victory_state(engine: Engine) -> None:
    """Minimal 'win' predicate for this simulator: alive, not lost, dungeon pile empty."""
    assert not engine.game_lost
    assert engine.health > 0
    assert len(engine.deck) == 0


# ---------------------------------------------------------------------------
# Game 1 — loss: survive room 1, die to a large monster in room 2
# ---------------------------------------------------------------------------


def test_full_game_loss_two_rooms_die_on_fourteenth_monster(make_engine):
    """
    Room 1: four 5s — resolve three fives (barehand) → health 5, foundation one 5.
    Room 2: [5 (foundation), 14, 2, 2] — resolving the 14 first kills the player.
    Deck order (draw order): four 5s, then 14, 2, 2.
    """
    deck = [
        C("spades", 5),
        C("clubs", 5),
        C("spades", 5),
        C("clubs", 5),
        C("spades", 14),
        C("spades", 2),
        C("clubs", 2),
    ]
    e = make_engine(deck=deck, health=20, weapon=(None, math.inf))
    assert e.health == 20
    assert not e.game_lost

    # --- Room 1 ---
    room1_cards = [deck[i] for i in range(4)]
    deal_room_from_deck(e)
    assert [e.hand[i] for i in range(4)] == room1_cards
    assert len(e.deck) == 3

    # Resolve three fives in order slots 0,1,2 (slot 3 stays as foundation)
    play_three_resolves(e, 0, 1, 2)
    assert e.health == 5
    assert not e.game_lost
    assert e.hand[3] is not None and e.hand[3].value == 5
    assert e.hand[0] is None and e.hand[1] is None and e.hand[2] is None

    foundation_r1 = e.hand[3]
    assert foundation_r1 is not None

    # --- Room 2 ---
    deal_room_with_foundation(e, foundation_r1)
    assert e.hand[0] is foundation_r1
    assert e.hand[1] is not None and e.hand[1].value == 14
    assert len(e.deck) == 0

    # Hit the 14 first (index 1) — instant loss from 5 HP
    assert not e.game_lost
    e.resolve_card(1)
    assert e.game_lost is True
    assert e.health <= 0


# ---------------------------------------------------------------------------
# Game 2 — win: weapon chain in room 1, potions in room 2, deck empty, alive
# ---------------------------------------------------------------------------


def test_full_game_win_two_rooms_exhaust_deck(make_engine):
    """
    Room 1: weapon 5, monsters 4,3,2 — equip, kill 4 (chain inf), kill 3 (3<4), foundation 2.
    Room 2: foundation monster 2 + three heart potions — drink three (only first heals at cap).
    Deck is exactly 7 cards so after dealing room 2 the draw pile is empty (3 potions + room1 = 7).
    """
    deck = [
        C("diamonds", 5),
        C("spades", 4),
        C("clubs", 3),
        C("spades", 2),
        C("hearts", 2),
        C("hearts", 2),
        C("hearts", 2),
    ]
    e = make_engine(deck=list(deck), health=20, weapon=(None, math.inf))

    # --- Room 1 ---
    deal_room_from_deck(e)
    w = e.hand[0]
    assert w is not None and w.type == "weapon" and w.value == 5

    e.resolve_card(0)
    assert e.weapon[0] is w
    assert e.weapon[1] == math.inf

    e.resolve_card(1)
    assert e.health == 20
    assert e.weapon[1] == 4

    e.resolve_card(2)
    assert e.health == 20
    assert e.weapon[1] == 3
    assert not e.game_lost

    foundation = e.hand[3]
    assert foundation is not None and foundation.value == 2

    # --- Room 2: three hearts left in deck → foundation + 3 potions, dungeon empty ---
    assert len(e.deck) == 3
    deal_room_with_foundation(e, foundation)
    assert e.hand[0] is foundation
    assert all(e.hand[i] is not None and e.hand[i].type == "potion" for i in (1, 2, 3))
    assert len(e.deck) == 0

    e.resolve_card(1)
    assert e.health == 20
    assert e.used_potion is True
    e.resolve_card(2)
    assert e.health == 20
    e.resolve_card(3)
    assert e.health == 20
    assert not e.game_lost

    # Unresolved foundation monster still in slot 0; draw pile is empty — good stopping point before RL.
    assert e.hand[0] is not None
    assert sum(1 for i in range(4) if e.hand[i] is not None) == 1

    assert_victory_state(e)


# ---------------------------------------------------------------------------
# Extra: single-room scripted loss (fast sanity)
# ---------------------------------------------------------------------------


def test_full_game_loss_one_room_three_resolves(make_engine):
    """Three monsters in one room: 8, 8, 8 — 20 - 8 - 8 = 4, third kill → -4."""
    deck = [
        C("spades", 8),
        C("clubs", 8),
        C("spades", 8),
        C("hearts", 2),
    ]
    e = make_engine(deck=deck)
    deal_room_from_deck(e)
    e.resolve_card(0)
    assert e.health == 12
    e.resolve_card(1)
    assert e.health == 4
    e.resolve_card(2)
    assert e.game_lost is True
    assert e.health < 0
