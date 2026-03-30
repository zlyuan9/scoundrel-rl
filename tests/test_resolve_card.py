"""
Branch tests for Engine.resolve_card aligned with DEVELOPER_GUIDE.md §2.6.

Weapon: damage max(0, monster - weapon); chain allows the next monster only if its rank is
≤ the last monster killed with that weapon. New weapon resets chain (last_slain = inf).

Potion: at most one effective potion per room turn; second in same turn has no effect.
"""

from __future__ import annotations

import math

import pytest

from scoundrel.card import Card


class _NonPlayableCard:
    """Not a Card; used to hit the final fallback branch in resolve_card."""

    type = "other"


# --- Monsters: barehand (no weapon) ---


def test_monster_barehand_no_weapon_reduces_health(make_engine):
    m = Card("spades", 7, "monster")
    e = make_engine(hand={0: m, 1: None, 2: None, 3: None}, health=20)
    e.resolve_card(0)
    assert e.health == 13
    assert e.game_lost is False
    assert e.hand[0] is None


def test_monster_barehand_no_weapon_game_lost_at_zero(make_engine):
    m = Card("clubs", 5, "monster")
    e = make_engine(hand={0: m, 1: None, 2: None, 3: None}, health=5)
    e.resolve_card(0)
    assert e.health == 0
    assert e.game_lost is True


def test_monster_barehand_no_weapon_game_lost_below_zero(make_engine):
    m = Card("spades", 10, "monster")
    e = make_engine(hand={0: m, 1: None, 2: None, 3: None}, health=5)
    e.resolve_card(0)
    assert e.health == -5
    assert e.game_lost is True


# --- Monsters: with weapon (first kill: last_slain is inf) ---


def test_monster_with_weapon_first_kill_damage_capped_at_zero(make_engine):
    w = Card("diamonds", 8, "weapon")
    m = Card("spades", 3, "monster")
    e = make_engine(
        hand={0: m, 1: None, 2: None, 3: None},
        health=20,
        weapon=(w, math.inf),
    )
    e.resolve_card(0)
    assert e.health == 20
    assert e.game_lost is False
    assert e.weapon == (w, 3)


def test_monster_with_weapon_partial_damage_survives(make_engine):
    w = Card("diamonds", 5, "weapon")
    m = Card("clubs", 11, "monster")
    e = make_engine(
        hand={0: m, 1: None, 2: None, 3: None},
        health=20,
        weapon=(w, math.inf),
    )
    e.resolve_card(0)
    assert e.health == 14
    assert e.game_lost is False
    assert e.weapon[1] == 11


def test_monster_with_weapon_damage_kills_player(make_engine):
    w = Card("diamonds", 2, "weapon")
    m = Card("spades", 14, "monster")
    e = make_engine(
        hand={0: m, 1: None, 2: None, 3: None},
        health=5,
        weapon=(w, math.inf),
    )
    e.resolve_card(0)
    assert e.health <= 0
    assert e.game_lost is True


# --- Weapon chain: second monster must be ≤ last kill (same rank allowed) ---


def test_monster_weapon_chain_second_weaker_uses_weapon(make_engine):
    w = Card("diamonds", 5, "weapon")
    e = make_engine(
        hand={0: Card("spades", 6, "monster"), 1: None, 2: None, 3: None},
        health=20,
        weapon=(w, 12),
    )
    e.resolve_card(0)
    assert e.health == 19
    assert e.weapon == (w, 6)


def test_monster_weapon_chain_equal_value_uses_weapon(make_engine):
    """Same rank as last kill (10) still allows weapon: damage max(0, 10-5)=5."""
    w = Card("diamonds", 5, "weapon")
    e = make_engine(
        hand={0: Card("spades", 10, "monster"), 1: None, 2: None, 3: None},
        health=20,
        weapon=(w, 10),
    )
    e.resolve_card(0)
    assert e.health == 15
    assert e.weapon == (w, 10)


def test_monster_weapon_chain_stronger_monster_forces_barehand(make_engine):
    w = Card("diamonds", 5, "weapon")
    e = make_engine(
        hand={0: Card("spades", 12, "monster"), 1: None, 2: None, 3: None},
        health=20,
        weapon=(w, 6),
    )
    e.resolve_card(0)
    assert e.health == 8
    assert e.weapon == (w, 6)


def test_monster_has_weapon_but_cannot_use_takes_full_damage(make_engine):
    """Equipped but chain blocks weapon → full monster value as barehand."""
    w = Card("diamonds", 9, "weapon")
    e = make_engine(
        hand={0: Card("clubs", 7, "monster"), 1: None, 2: None, 3: None},
        health=15,
        weapon=(w, 5),
    )
    e.resolve_card(0)
    assert e.health == 8
    assert e.weapon == (w, 5)


# --- Equip weapon ---


def test_weapon_equip_resets_chain_to_inf(make_engine):
    old = Card("diamonds", 4, "weapon")
    new_w = Card("diamonds", 9, "weapon")
    e = make_engine(
        hand={0: new_w, 1: None, 2: None, 3: None},
        weapon=(old, 7),
    )
    e.resolve_card(0)
    assert e.weapon[0] is new_w
    assert e.weapon[1] == math.inf


def test_weapon_downgrade_sets_info(make_engine):
    old = Card("diamonds", 9, "weapon")
    new_w = Card("diamonds", 4, "weapon")
    e = make_engine(
        hand={0: new_w, 1: None, 2: None, 3: None},
        weapon=(old, 7),
    )
    e.resolve_card(0)
    assert e.weapon[0] is new_w
    assert e._last_resolve_info.get("weapon_downgrade") is True


def test_weapon_first_equip_no_downgrade_info(make_engine):
    new_w = Card("diamonds", 4, "weapon")
    e = make_engine(
        hand={0: new_w, 1: None, 2: None, 3: None},
        weapon=(None, math.inf),
    )
    e.resolve_card(0)
    assert "weapon_downgrade" not in e._last_resolve_info


def test_weapon_upgrade_no_downgrade_info(make_engine):
    old = Card("diamonds", 4, "weapon")
    new_w = Card("diamonds", 9, "weapon")
    e = make_engine(
        hand={0: new_w, 1: None, 2: None, 3: None},
        weapon=(old, 7),
    )
    e.resolve_card(0)
    assert "weapon_downgrade" not in e._last_resolve_info


# --- Potions ---


def test_potion_heals_and_caps_at_20(make_engine):
    p = Card("hearts", 6, "potion")
    e = make_engine(hand={0: p, 1: None, 2: None, 3: None}, health=18)
    e.resolve_card(0)
    assert e.health == 20
    assert e.used_potion is True
    assert e._last_resolve_info.get("potion_overheal") == 4


def test_potion_heal_without_hitting_cap_branch(make_engine):
    """Covers potion branch where self.health > 20 is False after heal."""
    p = Card("hearts", 4, "potion")
    e = make_engine(hand={0: p, 1: None, 2: None, 3: None}, health=10)
    e.resolve_card(0)
    assert e.health == 14
    assert e.used_potion is True
    assert "potion_overheal" not in e._last_resolve_info


def test_potion_no_op_when_already_used_this_room(make_engine):
    p = Card("hearts", 5, "potion")
    e = make_engine(
        hand={0: p, 1: None, 2: None, 3: None},
        health=10,
        used_potion=True,
    )
    e.resolve_card(0)
    assert e.health == 10
    assert e.used_potion is True
    assert e._last_resolve_info.get("wasted_potion") is True


def test_potion_first_sets_flag_second_in_same_turn_no_heal(make_engine):
    """Rules: second potion in same turn is discarded (no heal)."""
    e = make_engine(
        hand={
            0: Card("hearts", 5, "potion"),
            1: Card("hearts", 8, "potion"),
            2: None,
            3: None,
        },
        health=10,
        used_potion=False,
    )
    e.resolve_card(0)
    assert e.health == 15
    assert e.used_potion is True
    e.resolve_card(1)
    assert e.health == 15
    assert e._last_resolve_info.get("wasted_potion") is True


# --- Unknown / fallback ---


def test_resolve_unknown_card_type_no_crash(make_engine):
    e = make_engine(hand={0: _NonPlayableCard(), 1: None, 2: None, 3: None}, health=20)
    e.resolve_card(0)
    assert e.health == 20
    assert e.game_lost is False
