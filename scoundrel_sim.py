"""
Monte Carlo simulator for the card game "Scoundrel".
Estimates the probability that a random deck is unwinnable under optimal play
using DFS with aggressive pruning.

Usage:
    python scoundrel_sim.py -n 10000 -w 8
"""

import random
import math
import multiprocessing as mp
from itertools import permutations
import argparse
import time

# ---------------------------------------------------------------------------
# Card encoding — use plain ints for speed
# ---------------------------------------------------------------------------
# Card = (type, value) where type: 0=monster, 1=weapon, 2=potion
MONSTER = 0
WEAPON = 1
POTION = 2
MAX_HP = 20


def build_deck() -> list[tuple[int, int]]:
    """Build the 44-card Scoundrel deck as (type, value) tuples."""
    deck = []
    for val in range(2, 15):  # Clubs 2-14
        deck.append((MONSTER, val))
    for val in range(2, 15):  # Spades 2-14
        deck.append((MONSTER, val))
    for val in range(2, 11):  # Diamonds 2-10
        deck.append((WEAPON, val))
    for val in range(2, 11):  # Hearts 2-10
        deck.append((POTION, val))
    assert len(deck) == 44
    return deck


# ---------------------------------------------------------------------------
# Upper-bound pruning helper
# ---------------------------------------------------------------------------

def can_possibly_survive(hp: int, remaining_cards: tuple, weapon_val: int) -> bool:
    """Quick check: can the player possibly survive all remaining monsters?
    Assumes best-case: always have the best weapon, heal fully with potions.
    This is a LOWER bound on damage, so if we still die, prune."""
    total_healing = 0
    best_weapon = weapon_val
    monsters = []
    for ctype, cval in remaining_cards:
        if ctype == POTION:
            total_healing += cval
        elif ctype == WEAPON:
            best_weapon = max(best_weapon, cval)
        else:
            monsters.append(cval)

    # Effective HP: current + all potions (capped loosely)
    effective_hp = min(hp + total_healing, MAX_HP + total_healing)

    # Minimum damage: each monster mitigated by best weapon
    min_damage = sum(max(0, m - best_weapon) for m in monsters)

    return effective_hp > min_damage


# ---------------------------------------------------------------------------
# Greedy heuristic — fast path for winnable decks
# ---------------------------------------------------------------------------

def greedy_solve(deck: tuple) -> bool:
    """Try a smart greedy strategy. If it survives, deck is winnable."""
    # Strategy: never avoid, leave behind least useful card, resolve in
    # smart order (weapon first if monsters present, potion before damage)
    hp = MAX_HP
    weapon_val = 0
    weapon_last = 0  # last monster slain with current weapon (0 = none)
    idx = 0
    leftover = None

    while True:
        # Build room
        cards_needed = 4 if leftover is None else 3
        if idx + cards_needed > len(deck):
            # Final room — resolve all remaining
            room = []
            if leftover is not None:
                room.append(leftover)
            room.extend(deck[idx:])
            hp, weapon_val, weapon_last = _greedy_resolve_all(
                hp, weapon_val, weapon_last, room)
            return hp > 0

        room = []
        if leftover is not None:
            room.append(leftover)
        room.extend(deck[idx:idx + cards_needed])
        idx += cards_needed

        # Choose which card to leave behind and resolve order
        best_hp = -1
        best_state = None
        for leave_i in range(4):
            left = room[leave_i]
            play = [room[j] for j in range(4) if j != leave_i]
            for perm in _unique_perms(play):
                h, wv, wl = _resolve_greedy(hp, weapon_val, weapon_last, perm)
                if h > best_hp:
                    best_hp = h
                    best_state = (h, wv, wl, left)

        if best_hp <= 0:
            return False
        hp, weapon_val, weapon_last, leftover = best_state

    return hp > 0


def _resolve_greedy(hp, weapon_val, weapon_last, cards):
    """Resolve cards greedily, return (hp, weapon_val, weapon_last)."""
    potion_used = False
    for ctype, cval in cards:
        if ctype == POTION:
            if not potion_used:
                hp = min(MAX_HP, hp + cval)
                potion_used = True
        elif ctype == WEAPON:
            weapon_val = cval
            weapon_last = 0
        else:  # monster
            if weapon_val > 0 and (weapon_last == 0 or cval <= weapon_last):
                hp -= max(0, cval - weapon_val)
                weapon_last = cval
            else:
                hp -= cval
            if hp <= 0:
                return (hp, weapon_val, weapon_last)
    return (hp, weapon_val, weapon_last)


def _greedy_resolve_all(hp, weapon_val, weapon_last, room):
    """Resolve all cards in a final room, trying all orderings."""
    best = (-999, 0, 0)
    for perm in _unique_perms(room):
        r = _resolve_greedy(hp, weapon_val, weapon_last, perm)
        if r[0] > best[0]:
            best = r
    return best


def _unique_perms(cards):
    """Generate permutations, deduplicating by (type, value) signature."""
    seen = set()
    for perm in permutations(cards):
        key = perm  # cards are already (type, val) tuples
        if key not in seen:
            seen.add(key)
            yield perm


# ---------------------------------------------------------------------------
# Full DFS solver
# ---------------------------------------------------------------------------

def solve(deck: tuple) -> bool:
    """Return True if there exists ANY legal decision path that survives."""
    # Fast path: try greedy
    if greedy_solve(deck):
        return True

    # Full DFS
    memo = {}
    return _dfs(MAX_HP, 0, 0, False, False, deck, None, memo)


def _dfs(hp: int, weapon_val: int, weapon_last: int,
         potion_used: bool, avoided_last: bool,
         dungeon: tuple, leftover, memo: dict) -> bool:
    """DFS over all legal decisions. Returns True if any path survives."""

    # Build room
    cards_needed = 4 if leftover is None else 3
    if len(dungeon) < cards_needed:
        # Final room
        room = []
        if leftover is not None:
            room.append(leftover)
        room.extend(dungeon)
        if not room:
            return True
        return _resolve_final(hp, weapon_val, weapon_last, potion_used, tuple(room), memo)

    if leftover is not None:
        room = (leftover,) + dungeon[:cards_needed]
    else:
        room = dungeon[:cards_needed]
    rest = dungeon[cards_needed:]

    # Memo key
    room_sorted = tuple(sorted(room))
    # For rest, we need the actual order (it determines future rooms)
    state_key = (hp, weapon_val, weapon_last, avoided_last, rest, room_sorted)
    cached = memo.get(state_key)
    if cached is not None:
        return cached

    # Upper-bound prune: can we possibly survive all remaining cards?
    all_remaining = room + rest
    if not can_possibly_survive(hp, all_remaining, weapon_val):
        memo[state_key] = False
        return False

    # --- Option 1: Avoid (if allowed) ---
    if not avoided_last:
        new_dungeon = rest + room
        if _dfs(hp, weapon_val, weapon_last, False, True,
                new_dungeon, None, memo):
            memo[state_key] = True
            return True

    # --- Option 2: Play 3 of 4 cards ---
    # Try each card as leftover, then try meaningful orderings of the other 3
    seen_configs = set()
    for leave_i in range(4):
        new_leftover = room[leave_i]
        play = tuple(room[j] for j in range(4) if j != leave_i)

        for perm in _unique_perms(play):
            # Resolve the 3 cards
            h, wv, wl, pu = _resolve_3(hp, weapon_val, weapon_last, perm)
            if h <= 0:
                continue

            # Dedup: same resulting state + same leftover + same rest = same future
            future_key = (h, wv, wl, new_leftover, pu)
            if future_key in seen_configs:
                continue
            seen_configs.add(future_key)

            if _dfs(h, wv, wl, pu, False, rest, new_leftover, memo):
                memo[state_key] = True
                return True

    memo[state_key] = False
    return False


def _resolve_3(hp, weapon_val, weapon_last, cards):
    """Resolve exactly 3 cards. Returns (hp, weapon_val, weapon_last, potion_used)."""
    potion_used = False
    for ctype, cval in cards:
        if ctype == POTION:
            if not potion_used:
                hp = min(MAX_HP, hp + cval)
                potion_used = True
        elif ctype == WEAPON:
            weapon_val = cval
            weapon_last = 0
        else:  # monster
            if weapon_val > 0 and (weapon_last == 0 or cval <= weapon_last):
                hp -= max(0, cval - weapon_val)
                weapon_last = cval
            else:
                hp -= cval
            if hp <= 0:
                return (hp, 0, 0, False)
    return (hp, weapon_val, weapon_last, potion_used)


def _resolve_final(hp, weapon_val, weapon_last, potion_used, room, memo):
    """Resolve final room — must play all cards, try all orderings."""
    key = ('F', hp, weapon_val, weapon_last, potion_used, tuple(sorted(room)))
    cached = memo.get(key)
    if cached is not None:
        return cached

    for perm in _unique_perms(room):
        h = hp
        wv = weapon_val
        wl = weapon_last
        pu = potion_used
        alive = True
        for ctype, cval in perm:
            if ctype == POTION:
                if not pu:
                    h = min(MAX_HP, h + cval)
                    pu = True
            elif ctype == WEAPON:
                wv = cval
                wl = 0
            else:
                if wv > 0 and (wl == 0 or cval <= wl):
                    h -= max(0, cval - wv)
                    wl = cval
                else:
                    h -= cval
                if h <= 0:
                    alive = False
                    break
        if alive and h > 0:
            memo[key] = True
            return True

    memo[key] = False
    return False


# ---------------------------------------------------------------------------
# Monte Carlo driver
# ---------------------------------------------------------------------------

def evaluate_deck(seed: int) -> bool:
    """Shuffle a deck with given seed, return True if unwinnable."""
    rng = random.Random(seed)
    deck = build_deck()
    rng.shuffle(deck)
    return not solve(tuple(deck))


def worker_batch(seeds: list[int]) -> int:
    """Process a batch of seeds. Returns count of unwinnable decks."""
    return sum(1 for s in seeds if evaluate_deck(s))


def run_simulation(n_trials: int, n_workers: int, batch_size: int = 50):
    """Run the Monte Carlo simulation with multiprocessing."""
    print(f"Scoundrel Monte Carlo Simulator")
    print(f"Trials: {n_trials}, Workers: {n_workers}, Batch size: {batch_size}")
    print()
    start = time.time()

    master_rng = random.Random(42)
    seeds = [master_rng.randint(0, 2**63) for _ in range(n_trials)]

    batches = [seeds[i:i + batch_size] for i in range(0, n_trials, batch_size)]

    unwinnable_total = 0
    batches_done = 0

    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_batch, batches):
            unwinnable_total += result
            batches_done += 1
            completed = min(batches_done * batch_size, n_trials)
            if completed % 500 < batch_size or completed >= n_trials:
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                p = unwinnable_total / completed
                print(f"  [{completed:>6}/{n_trials}] "
                      f"unwinnable={unwinnable_total} "
                      f"({p:.4f}) "
                      f"{rate:.1f} decks/s "
                      f"{elapsed:.1f}s")

    elapsed = time.time() - start
    p_hat = unwinnable_total / n_trials
    se = math.sqrt(p_hat * (1 - p_hat) / n_trials) if 0 < p_hat < 1 else 0
    ci_lo = max(0, p_hat - 1.96 * se)
    ci_hi = min(1, p_hat + 1.96 * se)

    print(f"\n{'='*60}")
    print(f"Results ({n_trials} trials, {elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Unwinnable decks:  {unwinnable_total} / {n_trials}")
    print(f"P(unwinnable):     {p_hat:.6f}")
    print(f"95% CI:            [{ci_lo:.6f}, {ci_hi:.6f}]")
    print(f"Throughput:        {n_trials / elapsed:.1f} decks/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation for Scoundrel unwinnability')
    parser.add_argument('-n', '--trials', type=int, default=10000,
                        help='Number of random decks to test (default: 10000)')
    parser.add_argument('-w', '--workers', type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help='Number of parallel workers')
    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Decks per worker batch (default: 50)')
    args = parser.parse_args()

    run_simulation(args.trials, args.workers, args.batch_size)
