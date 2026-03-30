#!/usr/bin/env python3
"""Plot training rollout reward and episode-length curves from TensorBoard run logs.

Reads ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean`` (not eval metrics). Stable-Baselines3
writes scalars under a directory that contains ``events.out.tfevents.*``. After training with
:mod:`scoundrel.train`, run::

    PYTHONPATH=. python analysis/plot_run.py
    PYTHONPATH=. python analysis/plot_run.py --logdir runs/20250101_120000/tensorboard/MaskablePPO_1
    PYTHONPATH=. python analysis/plot_run.py -o analysis/figures/my_plot.png
    PYTHONPATH=. python analysis/plot_run.py --note "PPO + shaping"
    echo "stdin note" | PYTHONPATH=. python analysis/plot_run.py --note -

Requires: ``pip install matplotlib tensorboard`` (often already present with SB3).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# TensorBoard ships EventAccumulator; matplotlib is optional for [analysis]
# ---------------------------------------------------------------------------


def _run_name_from_logdir(logdir: Path) -> str | None:
    """Infer timestamp folder under ``runs/`` (e.g. ``20260329_235557``) from a TensorBoard path."""
    cur = logdir.resolve()
    while cur.name and cur.parent != cur:
        if cur.name == "tensorboard":
            return cur.parent.name
        cur = cur.parent
    return None


def _default_figure_path(logdir: Path) -> Path:
    """``analysis/figures/<run_id>.png``, or ``plot_<timestamp>.png`` if run id unknown."""
    run = _run_name_from_logdir(logdir)
    if not run:
        stem = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in run)
    return Path("analysis/figures") / f"{stem}.png"


def _find_latest_events_parent(root: Path) -> Path:
    files = sorted(
        root.rglob("events.out.tfevents.*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files under {root}")
    return files[0].parent


def _load_scalars(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        events = ea.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in events]
    return out


def _plot(
    series: dict[str, list[tuple[int, float]]],
    out_path: Path,
    title: str | None,
    *,
    logdir: Path | None,
    note: str | None,
) -> None:
    import matplotlib.pyplot as plt

    reward_tags = ["rollout/ep_rew_mean"]
    reward_to_plot = [t for t in reward_tags if t in series and series[t]]
    if not reward_to_plot:
        available = list(series.keys())
        raise SystemExit(
            "No rollout/ep_rew_mean found. "
            f"Available scalar tags ({len(available)}): {available[:20]}"
            + (" …" if len(available) > 20 else "")
        )

    length_tags = ["rollout/ep_len_mean"]
    length_to_plot = [t for t in length_tags if t in series and series[t]]

    nrows = 2 if length_to_plot else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 4.5 * nrows + 0.8), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax0 = axes[0]
    for tag in reward_to_plot:
        steps, vals = zip(*series[tag])
        label = tag.replace("/", " — ")
        ax0.plot(steps, vals, marker="o", markersize=2, linewidth=1.2, label=label)
        last_s, last_v = steps[-1], vals[-1]
        ax0.annotate(
            f"last: {last_v:.4g} @ {last_s:,} steps",
            xy=(0.99, 0.02),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    ax0.set_ylabel("Reward")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left")
    if title:
        fig.suptitle(title)

    if length_to_plot:
        ax1 = axes[1]
        for tag in length_to_plot:
            steps, vals = zip(*series[tag])
            label = tag.replace("/", " — ")
            ax1.plot(steps, vals, marker="o", markersize=2, linewidth=1.2, label=label)
            last_s, last_v = steps[-1], vals[-1]
            ax1.annotate(
                f"last: {last_v:.4g} @ {last_s:,} steps",
                xy=(0.99, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.35",
            )
        ax1.set_xlabel("Environment steps (timesteps)")
        ax1.set_ylabel("Mean episode length")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left")
    else:
        ax0.set_xlabel("Environment steps (timesteps)")

    footer_parts: list[str] = []
    if logdir is not None:
        p = str(logdir.resolve())
        if len(p) > 96:
            p = "…" + p[-93:]
        footer_parts.append(f"Log: {p}")
    if note:
        footer_parts.append(f"Note: {note.strip()}")
    if footer_parts:
        footer = textwrap.fill("\n".join(footer_parts), width=110, break_long_words=False, break_on_hyphens=False)
        fig.text(
            0.02,
            0.015,
            footer,
            fontsize=7,
            va="bottom",
            ha="left",
            color="0.35",
            family="monospace",
        )

    foot = 0.18 if footer_parts else 0.07
    head = 0.93 if title else 0.97
    fig.tight_layout(rect=(0.03, foot, 0.97, head))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rollout reward and episode length from TensorBoard logs (training only)."
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="Directory containing TensorBoard event files (default: newest under ./runs)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: analysis/figures/<run_name>.png from --logdir)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (default: run name inferred from --logdir, e.g. 20260329_235557)",
    )
    parser.add_argument(
        "--note",
        "-n",
        type=str,
        default=None,
        metavar="TEXT",
        help="Optional comment printed in the figure footer. Use - to read from stdin.",
    )
    args = parser.parse_args()

    logdir = args.logdir
    if logdir is None:
        runs = Path("runs")
        if not runs.is_dir():
            raise SystemExit("No ./runs directory. Train with scoundrel.train first, or pass --logdir.")
        logdir = _find_latest_events_parent(runs)
        print(f"Using logdir: {logdir}")

    if args.note == "-":
        note = sys.stdin.read().strip() or None
    else:
        note = args.note

    out_path = args.output if args.output is not None else _default_figure_path(logdir)

    scalars = _load_scalars(logdir)
    title = args.title if args.title is not None else _run_name_from_logdir(logdir)
    _plot(scalars, out_path, title, logdir=logdir, note=note)


if __name__ == "__main__":
    main()
