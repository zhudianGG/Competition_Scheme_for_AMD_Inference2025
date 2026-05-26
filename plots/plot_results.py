"""Aggregate the five experiment CSVs into PNGs under plots/."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_e1(csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in df["draft_kind"].unique():
        sub = df[df["draft_kind"] == kind].sort_values("k")
        ax.plot(sub["k"], sub["accept_length"], marker="o", label=kind)
    ax.set_xlabel("draft K")
    ax.set_ylabel("avg accept length")
    ax.set_title("E1: Accept length vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "e1_accept_length.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4))
    base = df[(df["draft_kind"] == "none")]["throughput_tok_s"]
    base_v = float(base.iloc[0]) if len(base) else None
    for kind in df["draft_kind"].unique():
        if kind == "none":
            continue
        sub = df[df["draft_kind"] == kind].sort_values("k")
        speedup = sub["throughput_tok_s"] / base_v if base_v else sub["throughput_tok_s"]
        ax.bar([f"{kind}-K{k}" for k in sub["k"]], speedup)
    ax.set_ylabel("speedup vs no-spec")
    ax.set_title("E1: Throughput speedup")
    ax.axhline(1.0, color="k", linewidth=0.5)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "e1_speedup.png", dpi=150)


def plot_e2(csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in df["draft_kind"].unique():
        sub = df[df["draft_kind"] == kind].sort_values("prompt_len")
        ax.plot(sub["prompt_len"], sub["accept_length"], marker="o", label=kind)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("avg accept length")
    ax.set_title("E2: Accept length vs prompt length")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "e2_accept_decay.png", dpi=150)


def plot_e3(csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in df["draft_kind"].unique():
        sub = df[df["draft_kind"] == kind].sort_values("concurrency")
        ax.plot(sub["concurrency"], sub["throughput_tok_s"], marker="o", label=kind)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output throughput (tok/s)")
    ax.set_title("E3: Throughput vs concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "e3_throughput.png", dpi=150)


def plot_e4(csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for _, r in df.iterrows():
        by_cat[r["category"]].append(r["picked"] == "M=4")
    cats = sorted(by_cat.keys())
    rates = [sum(by_cat[c]) / len(by_cat[c]) * 100 for c in cats]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cats, rates)
    ax.set_ylabel("M=4 selection rate (%)")
    ax.set_title("E4: FourOverSix per-category split")
    ax.axhline(50, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "e4_m4_rate.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.results / "e1_accept_rate.csv").exists():
        plot_e1(args.results / "e1_accept_rate.csv", args.out)
    if (args.results / "e2_long_context.csv").exists():
        plot_e2(args.results / "e2_long_context.csv", args.out)
    if (args.results / "e3_concurrency.csv").exists():
        plot_e3(args.results / "e3_concurrency.csv", args.out)
    if (args.results / "e4_block_scale.csv").exists():
        plot_e4(args.results / "e4_block_scale.csv", args.out)
    print(f"[plots] wrote PNGs under {args.out}")


if __name__ == "__main__":
    main()
