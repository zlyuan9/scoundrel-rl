"""Load a saved MaskablePPO checkpoint and step through games in the pygame viewer.

Uses the same env stack as training (TimeLimit + ActionMasker). Arrow keys / Space
advance one policy step; Left undoes the last step (replays from episode seed); R
reshuffles a new episode; ESC quits.

Requires: pip install -e ".[gui,rl]"

Example::

    PYTHONPATH=. python -m scoundrel.replay_best_gui runs/20260330_004520
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pygame
from gymnasium.wrappers import TimeLimit
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from scoundrel.env import ScoundrelEnv
from viewer.main import GameView, infer_phase

# Keep in sync with scoundrel.train
MAX_EPISODE_STEPS = 512


def mask_fn(env) -> np.ndarray:
    return np.asarray(env.unwrapped.action_mask, dtype=np.bool_)


def make_masked_env():
    base = TimeLimit(ScoundrelEnv(), max_episode_steps=MAX_EPISODE_STEPS)
    return ActionMasker(base, mask_fn)


def _action_label(a: int) -> str:
    return ("skip", "enter", "slot0", "slot1", "slot2", "slot3")[a] if 0 <= a <= 5 else f"act{a}"


def resolve_model_zip(run_dir: Path) -> Path:
    """Prefer best_model from eval callback; fall back to final save."""
    candidates = (
        run_dir / "best_model" / "best_model.zip",
        run_dir / "maskable_ppo_final.zip",
    )
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"No model found in {run_dir}. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def find_latest_run_with_model() -> Path:
    root = Path("runs")
    if not root.is_dir():
        raise FileNotFoundError("No ./runs directory.")
    zips = list(root.glob("*/best_model/best_model.zip")) + list(root.glob("*/maskable_ppo_final.zip"))
    if not zips:
        raise FileNotFoundError("No best_model/best_model.zip or maskable_ppo_final.zip under runs/")
    return max(zips, key=lambda p: p.stat().st_mtime).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a trained MaskablePPO with step-through GUI.")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Training run directory (e.g. runs/20260329_235557). Default: newest under runs/ with a saved zip.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for env.reset (optional).")
    args = parser.parse_args()

    if args.run_dir is None:
        model_zip = find_latest_run_with_model()
    else:
        model_zip = resolve_model_zip(args.run_dir.resolve())

    print(f"Loading {model_zip}", flush=True)

    env = make_masked_env()
    model = MaskablePPO.load(str(model_zip), env=env)

    first_episode = True

    def next_episode_seed() -> int:
        nonlocal first_episode
        if first_episode:
            first_episode = False
            if args.seed is not None:
                return args.seed
        return random.randint(0, 2**31 - 1)

    episode_seed = next_episode_seed()
    action_history: list[int] = []
    obs, _ = env.reset(seed=episode_seed)
    log: list[str] = []
    episode = 1
    step_idx = 0
    pending_episode_end = False

    view = GameView()
    pygame.display.set_caption(f"Scoundrel RL replay — {model_zip.name}")

    def apply_history() -> None:
        """Rebuild env state from ``episode_seed`` and ``action_history`` (for undo)."""
        nonlocal obs, pending_episode_end, step_idx
        obs, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        for a in action_history:
            obs, _, terminated, truncated, _ = env.step(a)
        step_idx = len(action_history)
        pending_episode_end = terminated or truncated

    def redraw(msg: str) -> None:
        eng = env.unwrapped.engine
        ph = infer_phase(eng)
        view.render_engine(eng, phase=ph, message=msg, log=log)
        pygame.display.flip()

    redraw(
        f"Episode {episode} | step {step_idx} | "
        "→ ↓ Space: policy step | ←: undo | R: new shuffle | ESC: quit"
    )

    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit(0)
            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                env.close()
                pygame.quit()
                sys.exit(0)

            if event.key == pygame.K_r:
                episode += 1
                episode_seed = next_episode_seed()
                action_history = []
                log = []
                apply_history()
                redraw(f"Episode {episode} reset | → ↓ Space: step | ←: undo")
                continue

            if event.key == pygame.K_LEFT:
                if not action_history:
                    redraw(
                        f"Episode {episode} | step {step_idx} | Nothing to undo | "
                        "→ ↓ Space: step | ←: undo | R: shuffle"
                    )
                    continue
                action_history.pop()
                if log and log[-1] == "— episode end —":
                    log.pop()
                if log:
                    log.pop()
                apply_history()
                tail = ""
                eng = env.unwrapped.engine
                if eng.game_won:
                    tail = " | WON"
                elif eng.game_lost:
                    tail = " | LOST"
                elif pending_episode_end:
                    tail = " | TRUNCATED"
                redraw(
                    f"Episode {episode} | step {step_idx} (undone){tail} | "
                    + (
                        "→ ↓ Space: new episode"
                        if pending_episode_end
                        else "→ ↓ Space: step | ←: undo"
                    )
                )
                continue

            advance = event.key in (
                pygame.K_RIGHT,
                pygame.K_DOWN,
                pygame.K_SPACE,
            )
            if not advance:
                continue

            if pending_episode_end:
                episode += 1
                episode_seed = next_episode_seed()
                action_history = []
                log = []
                apply_history()
                redraw(f"Episode {episode} (new run) | → ↓ Space: step | ←: undo")
                continue

            masks = get_action_masks(env)
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=masks,
            )
            act = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(act)
            action_history.append(act)
            step_idx = len(action_history)

            line = f"step {step_idx}: {_action_label(act)} (a={act}) r={reward:.3g}"
            if info.get("invalid_action"):
                line += " [INVALID]"
            log.append(line)
            log = log[-6:]

            eng = env.unwrapped.engine
            tail = ""
            if eng.game_won:
                tail = " | WON"
            elif eng.game_lost:
                tail = " | LOST"
            elif truncated:
                tail = " | TRUNCATED"

            pending_episode_end = terminated or truncated

            redraw(
                f"Episode {episode} | {line}{tail} | "
                + (
                    "→ ↓ Space: new episode | ←: undo last step"
                    if pending_episode_end
                    else "→ ↓ Space: step | ←: undo | R: shuffle"
                )
            )

            if pending_episode_end:
                log.append("— episode end —")

        clock.tick(60)

    env.close()


if __name__ == "__main__":
    main()
