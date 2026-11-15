"""Figures for the slides and the report (PNG, written to report/figures/).
Reads the result tables in outputs/, so run the pipeline first."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import macro as mc
from .config import ROOT, path
from .panel import DEFAULT, load_panel

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
HIST = "#52514e"
NAMES = {"baseline": "Baseline", "severely_adverse": "Severely adverse"}
COLORS = {"baseline": BLUE, "severely_adverse": ORANGE}

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "font.family": "sans-serif",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "lines.linewidth": 2, "legend.frameon": False,
})


def _save(fig, name):
    out = ROOT / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name)
    plt.close(fig)
    print(f"wrote report/figures/{name}")


def _windows(ax, as_of, cfg):
    """Shade the forecast window and the reversion period."""
    f, r = cfg["ecl"]["forecast_months"], cfg["ecl"]["reversion_months"]
    end_f = as_of + pd.DateOffset(months=f)
    ax.axvspan(as_of, end_f, color="#000000", alpha=0.05, lw=0)
    ax.axvspan(end_f, end_f + pd.DateOffset(months=r), color="#000000", alpha=0.025, lw=0)


def vintage_defaults(cfg):
    p = load_panel(path(cfg, "processed"), columns=["loan_id", "event", "vintage"])
    g = p.groupby("vintage").agg(loans=("loan_id", "nunique"), defaults=("event", lambda e: (e == DEFAULT).sum()))
    rate = 100 * g["defaults"] / g["loans"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(rate.index, rate.values, color=BLUE, width=0.7)
    for v in [2005, 2006, 2007, 2008]:
        ax.text(v, rate[v] + 0.25, f"{rate[v]:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_title("Share of loans that defaulted, by origination year")
    ax.set_ylabel("% of loans")
    ax.set_xticks(rate.index[::2])
    ax.text(1, -0.16, "Recent years have had less time to default. Sample: 5,000 loans per year.",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    _save(fig, "vintage_defaults.png")


def scenarios(cfg):
    out = path(cfg, "outputs")
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    hist = mc.load_macro(cfg).loc["2018-01-01":as_of]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in NAMES:
        pth = pd.read_csv(out / f"macro_path_{name}.csv", index_col=0, parse_dates=True).loc[:"2030-12-01"]
        axes[0].plot(pth.index, pth["unrate"], color=COLORS[name], label=NAMES[name])
        axes[1].plot(pth.index, 100 * pth["hpi"] / hist["hpi"].iloc[-1], color=COLORS[name], label=NAMES[name])
    axes[0].plot(hist.index, hist["unrate"], color=HIST, label="History")
    axes[1].plot(hist.index, 100 * hist["hpi"] / hist["hpi"].iloc[-1], color=HIST, label="History")
    axes[0].set_title("Unemployment rate (%)")
    axes[1].set_title("House prices (Dec 2025 = 100)")
    for ax in axes:
        _windows(ax, as_of, cfg)
    axes[1].legend(loc="upper left")
    fig.text(0.99, -0.02, f"Shaded: {cfg['ecl']['forecast_months']}-month Fed scenario, then "
             f"{cfg['ecl']['reversion_months']}-month reversion to long-run averages.",
             ha="right", fontsize=9, color=MUTED)
    _save(fig, "scenarios.png")


def backtest(cfg):
    b = pd.read_csv(path(cfg, "outputs") / "backtest_by_month.csv", index_col=0, parse_dates=True)
    q = 100 * b.resample("QE").mean()
    q.index = q.index.to_period("Q").to_timestamp()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, actual, pred, title in [(axes[0], "default", "p_default", "Monthly default rate (%)"),
                                    (axes[1], "prepay", "p_prepay", "Monthly prepayment rate (%)")]:
        ax.plot(q.index, q[actual], color=BLUE, label="Actual")
        ax.plot(q.index, q[pred], color=ORANGE, label="Predicted")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes[1].legend(loc="upper right")
    fig.text(0.99, -0.02, f"Model fitted on 2005-{cfg['pd']['train_end'][:4]} data only; quarterly averages.",
             ha="right", fontsize=9, color=MUTED)
    _save(fig, "backtest.png")


def reserve(cfg):
    out = path(cfg, "outputs")
    s = pd.read_csv(out / "ecl_summary.csv", index_col=0)
    d = pd.read_csv(out / "dfast_comparison.csv", index_col=0)["value"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = [NAMES[n] for n in s.index]
    axes[0].bar(x, s["reserve_rate_pct"], color=[COLORS[n] for n in s.index], width=0.55)
    for i, v in enumerate(s["reserve_rate_pct"]):
        axes[0].text(i, v + 0.03, f"{v:.2f}%", ha="center", color=INK)
    axes[0].set_title("Lifetime reserve (% of balance)")
    bars = {"Our model": d["model_9q_loss_rate_pct"], "Fed DFAST 2026": d["published_9q_loss_rate_pct"]}
    axes[1].bar(list(bars), list(bars.values()), color=[ORANGE, HIST], width=0.55)
    for i, v in enumerate(bars.values()):
        axes[1].text(i, v + 0.03, f"{v:.2f}%", ha="center", color=INK)
    axes[1].set_title("9-quarter loss rate, severely adverse")
    for ax in axes:
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.8))
    _save(fig, "reserve.png")


def loss_timing(cfg):
    out = path(cfg, "outputs")
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in NAMES:
        m = pd.read_csv(out / f"ecl_monthly_{name}.csv", index_col=0, parse_dates=True).iloc[:120]
        ax.plot(m.index, m["loss"] / 1e6, color=COLORS[name], label=NAMES[name])
    _windows(ax, as_of, cfg)
    ax.set_title("Expected credit loss by month ($ millions)")
    ax.legend(loc="upper right")
    ax.text(1, -0.14, "Shaded: forecast window, then reversion. First 10 years shown.",
            transform=ax.transAxes, ha="right", fontsize=9, color=MUTED)
    _save(fig, "loss_timing.png")


def lgd(cfg):
    t = pd.read_csv(path(cfg, "outputs") / "lgd_table.csv")
    labels = [f"<{int(t.cltv_to[0])}"] + [f"{int(a)}-{int(b)}" for a, b in zip(t.cltv_from[1:-1], t.cltv_to[1:-1])] \
        + [f">{int(t.cltv_from.iloc[-1])}"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, 100 * t["severity"], color=BLUE, width=0.6)
    for i, (v, n) in enumerate(zip(100 * t["severity"], t["n"])):
        ax.text(i, v + 0.8, f"{v:.0f}%", ha="center", color=INK, fontsize=10)
        ax.text(i, -4.5, f"n={n}", ha="center", color=MUTED, fontsize=8)
    ax.set_title("Loss severity by current LTV at default")
    ax.set_xlabel("Current LTV (%)", labelpad=14)
    ax.set_ylabel("% of balance at default")
    _save(fig, "lgd.png")


def make_all(cfg):
    for f in [vintage_defaults, scenarios, backtest, reserve, loss_timing, lgd]:
        f(cfg)
