from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from scoundrel.env import ScoundrelEnv
from scoundrel.deck_stats_callback import DeckStatsTensorboardCallback
from scoundrel.win_rate_callback import WinRateTensorboardCallback

GAMMA = 0.99

# Truncate episodes after this many env steps (win/lose still ends earlier).
MAX_EPISODE_STEPS = 512

LOG_INTERVAL = 5
N_TRAIN_ENVS = 8
TOTAL_TIMESTEPS = 20_000_000

# ---------------------------------------------------------------------------
# Reward configuration — all passed explicitly to ScoundrelEnv
# ---------------------------------------------------------------------------
SPARSE_REWARD_ONLY = False
WIN_REWARD = 5.0           # rare wins need strong signal early in training
LOSS_REWARD = -1.0         # reduced from -2 so dense shaping isn't drowned out
ROOM_SURVIVED_BONUS = 0.15 # clear per-room progress signal
DECK_DAMAGE_LEAVE_REWARD = 0.003
ROOM_HP_LOSS_PENALTY_PER_HP = 0.01
POTENTIAL_SHAPING_SCALE = 0.3
WASTED_POTION_PENALTY = 0.05
POTION_OVERHEAL_PENALTY_PER_HP = 0.01
WEAPON_DOWNGRADE_PENALTY = 0.0  # sometimes correct play; don't penalise


def mask_fn(env: gym.Env) -> np.ndarray:
    return np.asarray(env.unwrapped.action_mask, dtype=np.bool_)


def make_masked_env() -> gym.Env:
    base = TimeLimit(
        ScoundrelEnv(
            sparse_reward_only=SPARSE_REWARD_ONLY,
            win_reward=WIN_REWARD,
            loss_reward=LOSS_REWARD,
            room_survived_bonus=ROOM_SURVIVED_BONUS,
            deck_damage_leave_reward=DECK_DAMAGE_LEAVE_REWARD,
            room_hp_loss_penalty_per_hp=ROOM_HP_LOSS_PENALTY_PER_HP,
            potential_shaping_gamma=GAMMA,
            potential_shaping_scale=POTENTIAL_SHAPING_SCALE,
            wasted_potion_penalty=WASTED_POTION_PENALTY,
            potion_overheal_penalty_per_hp=POTION_OVERHEAL_PENALTY_PER_HP,
            weapon_downgrade_penalty=WEAPON_DOWNGRADE_PENALTY,
        ),
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    return ActionMasker(base, mask_fn)


def linear_schedule(initial_value: float):
    """Linear decay from ``initial_value`` to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def _tensorboard_events_dir(tb_log: Path) -> Path:
    events = list(tb_log.rglob("events.out.tfevents.*"))
    if events:
        return max(events, key=lambda p: p.stat().st_mtime).parent
    return tb_log


def main() -> None:
    run_root = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    tb_log = run_root / "tensorboard"

    train_env = make_vec_env(
        make_masked_env,
        n_envs=N_TRAIN_ENVS,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = make_masked_env()

    eval_freq = 10_000
    eval_callback = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=20,
        best_model_save_path=str(run_root / "best_model"),
        log_path=str(run_root / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    win_rate_callback = WinRateTensorboardCallback(window_size=100)
    deck_stats_callback = DeckStatsTensorboardCallback()

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=linear_schedule(3e-4),
        n_steps=2048,
        batch_size=256,
        gamma=GAMMA,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(tb_log),
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, win_rate_callback, deck_stats_callback],
        progress_bar=True,
        log_interval=LOG_INTERVAL,
    )
    model.save(run_root / "maskable_ppo_final")

    train_env.close()
    eval_env.close()

    plot_logdir = _tensorboard_events_dir(tb_log).resolve()
    try:
        from analysis.plot_run import write_training_figure

        fig_path = write_training_figure(plot_logdir)
        print(f"Figure: {fig_path}")
    except ImportError as e:
        print(
            "Plot skipped (need matplotlib + tensorboard; "
            f"pip install -e '.[analysis]'): {e}"
        )
    except SystemExit as e:
        msg = e.args[0] if e.args else e
        print(f"Plot skipped: {msg}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    plot_cmd = f"PYTHONPATH=. python analysis/plot_run.py --logdir {plot_logdir}"
    replay_cmd = f"PYTHONPATH=. python -m scoundrel.replay_best_gui {run_root.resolve()}"
    print(f"Run directory: {run_root.resolve()}")
    print(f"TensorBoard: tensorboard --logdir {tb_log.resolve()}")
    print(f"Plot (manual): {plot_cmd}")
    print(f"Replay (manual): {replay_cmd}")


if __name__ == "__main__":
    main()
