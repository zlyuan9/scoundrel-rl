"""Tests for create_deck, Engine construction, and RL-style step flow (no stdin in Engine)."""

from __future__ import annotations

import random

import pytest

from scoundrel.card import Card
from scoundrel.engine import Engine, create_deck


def test_create_deck_has_44_cards():
    deck = create_deck()
    assert len(deck) == 44


def test_engine_init_shuffles_and_has_full_deck():
    random.seed(42)
    e = Engine()
    assert len(e.deck) == 44
    assert e.health == 20
    assert e.game_lost is False
    assert e.game_won is False
    assert e.used_potion is False
    assert e.skipped_room is False


def test_str_and_repr_contain_deck():
    e = Engine()
    assert "Engine(deck=" in str(e)
    assert "Engine(deck=" in repr(e)


def test_get_deck_returns_same_list():
    e = Engine()
    assert e.get_deck() is e.deck


def test_play_room_calls_resolve_three_times(make_engine):
    e = make_engine(
        hand={
            0: Card("spades", 2, "monster"),
            1: Card("clubs", 3, "monster"),
            2: Card("spades", 4, "monster"),
            3: Card("hearts", 5, "potion"),
        },
        health=20,
    )
    e.begin_play_room()
    e.resolve_in_room(0)
    e.resolve_in_room(1)
    e.resolve_in_room(2)
    assert e.hand[0] is None and e.hand[1] is None and e.hand[2] is None
    assert e.hand[3] is not None


def test_play_turn_play_room_integration():
    """Deal first room, enter, resolve three cards; room_done then start_turn() deals next room."""
    random.seed(0)
    e = Engine()
    e.start_turn()
    e.step(1)
    e.step(2)
    e.step(3)
    e.step(4)
    assert e._resolves_left == 0
    assert not e.game_lost
    assert all(e.hand[i] is not None for i in range(4))


def test_play_turn_invalid_then_recurse():
    random.seed(0)
    e = Engine()
    e.start_turn()
    _, _, _, info = e.step(5)
    assert info.get("invalid_action") is True
    e.step(1)
    e.step(2)
    e.step(3)
    e.step(4)
    assert e._resolves_left == 0
    assert not e.game_lost


def test_play_turn_skip_then_play():
    """Skip then enter room and resolve three picks (outcome depends on shuffle)."""
    random.seed(0)
    e = Engine()
    e.start_turn()
    e.step(0)
    e.step(1)
    e.step(2)
    e.step(3)
    e.step(4)
    assert e._resolves_left == 0 or e.game_lost or e.game_won


def test_play_turn_skip_blocked_then_play():
    """Second skip while blocked → forced into room (matches viewer)."""
    random.seed(0)
    e = Engine()
    e.start_turn()
    e.step(0)
    _, _, _, info = e.step(0)
    assert "invalid_action" not in info
    e.step(2)
    e.step(3)
    e.step(4)
    assert e._resolves_left == 0 or e.game_lost or e.game_won


def test_fill_room_first_room_deals_four(make_engine):
    deck = [Card("spades", i, "monster") for i in range(2, 6)]
    e = make_engine(deck=deck)
    e._fill_room()
    assert len(e.deck) == 0
    assert all(e.hand[i] is not None for i in range(4))


def test_fill_room_not_enough_cards_sets_won(make_engine):
    e = make_engine(deck=[Card("spades", 2, "monster")] * 3)
    e._fill_room()
    assert e.game_won


def test_fill_room_foundation_plus_three_draws(make_engine):
    foundation = Card("spades", 7, "monster")
    e = make_engine(
        deck=[
            Card("hearts", 2, "potion"),
            Card("hearts", 3, "potion"),
            Card("hearts", 4, "potion"),
        ],
        hand={0: foundation, 1: None, 2: None, 3: None},
    )
    e._fill_room()
    assert e.hand[0] is foundation
    assert e.hand[1].type == "potion"


def test_fill_room_foundation_deck_short_sets_won(make_engine):
    foundation = Card("spades", 7, "monster")
    e = make_engine(
        deck=[Card("hearts", 2, "potion"), Card("hearts", 3, "potion")],
        hand={0: foundation, 1: None, 2: None, 3: None},
    )
    e._fill_room()
    assert e.game_won


def test_fill_room_already_four_is_noop(make_engine):
    cards = [Card("spades", i, "monster") for i in range(2, 6)]
    e = make_engine(
        deck=[Card("hearts", 2, "potion")] * 10,
        hand={i: cards[i] for i in range(4)},
    )
    before = len(e.deck)
    e._fill_room()
    assert len(e.deck) == before


def test_fill_room_invalid_card_count_raises(make_engine):
    e = make_engine(
        hand={
            0: Card("spades", 2, "monster"),
            1: Card("spades", 3, "monster"),
            2: None,
            3: None,
        },
        deck=[],
    )
    with pytest.raises(RuntimeError, match="invalid hand state"):
        e._fill_room()


def test_start_turn_won_empty_deck(make_engine):
    e = make_engine(deck=[])
    e.start_turn()
    assert e.game_won


def test_try_skip_returns_false_when_double_skip_blocked(make_engine):
    e = make_engine(
        deck=[Card("spades", 2, "monster")] * 8,
        hand={0: Card("spades", 3, "monster"), 1: None, 2: None, 3: None},
        skipped_room=True,
    )
    assert e.try_skip() is False


def test_resolve_in_room_without_begin_raises(make_engine):
    e = make_engine(
        hand={
            0: Card("spades", 2, "monster"),
            1: None,
            2: None,
            3: None,
        },
        deck=[Card("hearts", 2, "potion")] * 10,
    )
    with pytest.raises(RuntimeError, match="begin_play_room"):
        e.resolve_in_room(0)


def test_step_invalid_empty_slot_in_room(make_engine):
    e = make_engine(
        hand={
            0: Card("spades", 2, "monster"),
            1: None,
            2: None,
            3: None,
        },
        deck=[Card("hearts", 2, "potion")] * 10,
    )
    e.begin_play_room()
    _, _, _, info = e.step(4)
    assert info.get("invalid_action") is True
