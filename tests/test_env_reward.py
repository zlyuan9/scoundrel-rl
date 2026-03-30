"""Reward: terminal win/loss, deck damage bonus, room HP-loss penalty."""

import pytest

from scoundrel.card import Card
from scoundrel.env import (
    DEFAULT_LOSS_REWARD,
    DEFAULT_ROOM_HP_LOSS_PENALTY_PER_HP,
    TOTAL_MONSTER_CARDS,
    ScoundrelEnv,
    _count_monsters_in_deck_and_hand,
    _monster_damage_sum_in_deck,
    _monster_progress_potential,
)


def test_monsters_in_play_total_26_after_reset():
    env = ScoundrelEnv()
    env.reset(seed=42)
    assert _count_monsters_in_deck_and_hand(env.engine) == TOTAL_MONSTER_CARDS


def test_monster_progress_potential_zero_when_all_26_in_play():
    env = ScoundrelEnv()
    env.reset(seed=42)
    assert _monster_progress_potential(env.engine) == pytest.approx(0.0)
    assert env._prev_phi == pytest.approx(0.0)


def test_loss_reward_passed_to_engine():
    env = ScoundrelEnv(loss_reward=-2.0)
    env.reset(seed=0)
    assert env.engine._loss_reward == -2.0
    env.engine.game_lost = True
    assert env.engine._reward_for_terminal() == -2.0


def test_prev_deck_monster_damage_tracks_after_valid_step():
    env = ScoundrelEnv()
    env.reset(seed=42)
    env.step(1)
    assert env._prev_deck_monster_damage == _monster_damage_sum_in_deck(env.engine)


def test_enter_room_sets_health_at_room_start():
    env = ScoundrelEnv()
    env.reset(seed=42)
    assert env._health_at_room_start is None
    env.step(1)
    assert env._health_at_room_start == float(env.engine.health)


def test_room_hp_loss_penalty_on_death_in_room():
    env = ScoundrelEnv(sparse_reward_only=False, potential_shaping=False,
                       loss_reward=-2.0)
    env.reset(seed=0)
    e = env.engine
    e.hand = {0: Card("spades", 10, "monster"), 1: None, 2: None, 3: None}
    e.deck = []
    e.health = 5
    e.weapon = (None, float("inf"))
    e.used_potion = False
    e._resolves_left = 3
    e.game_won = False
    e.game_lost = False
    env._prev_deck_monster_damage = _monster_damage_sum_in_deck(e)
    env._health_at_room_start = 20.0

    _obs, r, terminated, _, _info = env.step(2)
    assert e.game_lost
    assert terminated
    expected_penalty = DEFAULT_ROOM_HP_LOSS_PENALTY_PER_HP * (20.0 - 0.0)
    assert r == pytest.approx(-2.0 - expected_penalty)


def test_sparse_reward_only_skips_room_penalty():
    env = ScoundrelEnv(sparse_reward_only=True, loss_reward=-2.0)
    env.reset(seed=0)
    e = env.engine
    e.hand = {0: Card("spades", 10, "monster"), 1: None, 2: None, 3: None}
    e.deck = []
    e.health = 5
    e.weapon = (None, float("inf"))
    e.used_potion = False
    e._resolves_left = 3
    e.game_won = False
    e.game_lost = False
    env._prev_deck_monster_damage = _monster_damage_sum_in_deck(e)
    env._health_at_room_start = 20.0

    _obs, r, terminated, _, _info = env.step(2)
    assert e.game_lost and terminated
    assert r == pytest.approx(-2.0)
