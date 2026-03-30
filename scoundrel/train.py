from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from scoundrel.env import DEFAULT_POTENTIAL_SHAPING_GAMMA, ScoundrelEnv

# Match PPO ``gamma`` to potential-based shaping in :class:`ScoundrelEnv` (Ng et al.).
GAMMA = DEFAULT_POTENTIAL_SHAPING_GAMMA

# Truncate episodes after this many env steps (win/lose still ends earlier).
# Normal games are usually much shorter; this caps pathological long runs.
MAX_EPISODE_STEPS = 512

# Rollout iterations between training logs (console + rollout/* TensorBoard scalars).
# SB3 default is 1 (log every PPO update). Increase to print less often.
LOG_INTERVAL = 5


def mask_fn(env: gym.Env) -> np.ndarray:
    return np.asarray(env.unwrapped.action_mask, dtype=np.bool_)


def make_masked_env() -> gym.Env:
    base = TimeLimit(
        ScoundrelEnv(potential_shaping_gamma=GAMMA),
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return ActionMasker(base, mask_fn)


def _tensorboard_events_dir(tb_log: Path) -> Path:
    """Directory containing event files (e.g. .../tensorboard/MaskablePPO_1)."""
    events = list(tb_log.rglob("events.out.tfevents.*"))
    if events:
        return max(events, key=lambda p: p.stat().st_mtime).parent
    return tb_log


def main() -> None:
    run_root = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    tb_log = run_root / "tensorboard"

    train_env = make_masked_env()
    eval_env = make_masked_env()

    # eval_freq in *environment* timesteps between evaluations
    eval_freq = 5000
    # Must use MaskableEvalCallback: plain EvalCallback does not pass action masks,
    # so eval picks illegal actions (no game progress, 0 reward until TimeLimit=512).
    eval_callback = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=5,
        best_model_save_path=str(run_root / "best_model"),
        log_path=str(run_root / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=GAMMA,
        tensorboard_log=str(tb_log),
    )
    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
        progress_bar=True,
        log_interval=LOG_INTERVAL,
    )
    model.save(run_root / "maskable_ppo_final")

    train_env.close()
    eval_env.close()

    plot_logdir = _tensorboard_events_dir(tb_log).resolve()
    plot_cmd = f"PYTHONPATH=. python analysis/plot_run.py --logdir {plot_logdir}"
    print(f"Run directory: {run_root.resolve()}")
    print(f"TensorBoard: tensorboard --logdir {tb_log.resolve()}")
    print(f"Plot: {plot_cmd}")


if __name__ == "__main__":
    main()
