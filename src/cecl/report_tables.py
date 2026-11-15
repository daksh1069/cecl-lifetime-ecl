"""LaTeX tables for the report (report/tables/*.tex), built from the pipeline
outputs so no number in the report is typed by hand."""
import numpy as np
import pandas as pd

from .config import ROOT, path
from .panel import CURRENT, DEFAULT, PREPAY, load_panel

OUT = ROOT / "report" / "tables"

FEATURE_LABELS = {
    "const": "Constant",
    "fico": "FICO (per 50 points above 740)",
    "dti": "DTI (per 10 points above 35)",
    "cltv": "Current LTV (per 10 points above 70)",
    "cltv_over_100": "Current LTV above 100 (per 10 points)",
    "log_age": "log(loan age, months)",
    "age_years": "Loan age (years)",
    "incentive": "Refinance incentive, floored at $-0.5$ (pct pts)",
    "incentive_pos": "Positive part of incentive (pct pts)",
    "unrate": "Unemployment rate (\\%)",
    "hpi_yoy": "House price growth, year on year (\\%)",
}


def _write(name: str, header: str, rows: list[str], align: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = "\n".join(rows)
    tex = f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
    (OUT / name).write_text(tex)
    print(f"wrote report/tables/{name}")


def _stars(p: float) -> str:
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def sample_table(cfg) -> None:
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    p = load_panel(path(cfg, "processed"), columns=["loan_id", "period", "event", "vintage"])
    p = p[p["period"] <= as_of]
    g = p.groupby("vintage").agg(loans=("loan_id", "nunique"), months=("event", "size"),
                                 defaults=("event", lambda e: (e == DEFAULT).sum()),
                                 prepays=("event", lambda e: (e == PREPAY).sum()))
    rows = [f"{v} & {r.loans:,} & {r.months:,} & {r.defaults:,} & {r.prepays:,} & {100 * r.defaults / r.loans:.1f} \\\\"
            for v, r in g.iterrows()]
    t = g.sum()
    rows.append("\\midrule")
    rows.append(f"Total & {t.loans:,} & {t.months:,} & {t.defaults:,} & {t.prepays:,} & {100 * t.defaults / t.loans:.1f} \\\\")
    _write("sample.tex", "Vintage & Loans & Loan-months & Defaults & Prepayments & Default \\%", rows, "lrrrrr")


def pd_table(cfg) -> None:
    t = pd.read_csv(path(cfg, "outputs") / "pd_coefficients.csv", header=[0, 1], index_col=0)
    rows = []
    for feat in t.index:
        cells = []
        for col in ["default", "prepay"]:
            c, s, p = t[("coef", col)][feat], t[("se", col)][feat], t[("p", col)][feat]
            cells.append(f"${c:.3f}{_stars(p)}$ & $({s:.3f})$")
        rows.append(f"{FEATURE_LABELS.get(feat, feat)} & {' & '.join(cells)} \\\\")
    _write("pd_coefficients.tex",
           "Variable & \\multicolumn{2}{c}{Default} & \\multicolumn{2}{c}{Prepayment}", rows, "lrrrr")


def lgd_table(cfg) -> None:
    t = pd.read_csv(path(cfg, "outputs") / "lgd_table.csv")
    rows = []
    for _, r in t.iterrows():
        lo, hi = int(r.cltv_from), r.cltv_to
        label = f"Below {int(hi)}" if lo == 0 else (f"Above {lo}" if np.isinf(hi) else f"{lo}--{int(hi)}")
        rows.append(f"{label} & {int(r.n):,} & {100 * r.severity:.1f} \\\\")
    _write("lgd.tex", "Current LTV (\\%) & Resolved defaults & Severity (\\%)", rows, "lrr")


def ecl_table(cfg) -> None:
    s = pd.read_csv(path(cfg, "outputs") / "ecl_summary.csv", index_col=0)
    b, a = s.loc["baseline"], s.loc["severely_adverse"]
    lines = [
        ("Loans open at the reserve date", f"{int(b.loans):,}", f"{int(a.loans):,}"),
        ("Unpaid balance (\\$ billion)", f"{b.upb / 1e9:.2f}", f"{a.upb / 1e9:.2f}"),
        ("Lifetime ECL, discounted (\\$ million)", f"{b.ecl / 1e6:.1f}", f"{a.ecl / 1e6:.1f}"),
        ("Reserve rate (\\% of balance)", f"{b.reserve_rate_pct:.2f}", f"{a.reserve_rate_pct:.2f}"),
        ("Lifetime loss, undiscounted (\\%)", f"{b.lifetime_loss_undiscounted_pct:.2f}", f"{a.lifetime_loss_undiscounted_pct:.2f}"),
        ("Loss in the first 9 quarters (\\%)", f"{b.loss_rate_9q_pct:.2f}", f"{a.loss_rate_9q_pct:.2f}"),
        ("Balance expected to default (\\%)", f"{b.lifetime_default_pct:.2f}", f"{a.lifetime_default_pct:.2f}"),
    ]
    _write("ecl.tex", "& Baseline & Severely adverse", [f"{l} & {x} & {y} \\\\" for l, x, y in lines], "lrr")


def backtest_table(cfg) -> None:
    m = pd.read_csv(path(cfg, "outputs") / "backtest_metrics.csv", index_col=0)["value"]
    f = lambda k: float(m[k])
    lines = [
        ("Training loan-months (through " + m["train_end"] + ")", f"{int(f('train_rows')):,}", ""),
        ("Test loan-months (after " + m["train_end"] + ")", f"{int(f('test_rows')):,}", ""),
        ("AUC, out of time", f"{f('auc_default'):.3f}", f"{f('auc_prepay'):.3f}"),
        ("Actual monthly rate (\\%)", f"{f('actual_default_rate_pct'):.3f}", f"{f('actual_prepay_rate_pct'):.3f}"),
        ("Predicted monthly rate (\\%)", f"{f('predicted_default_rate_pct'):.3f}", f"{f('predicted_prepay_rate_pct'):.3f}"),
    ]
    _write("backtest.tex", "& Default & Prepayment", [f"{l} & {x} & {y} \\\\" for l, x, y in lines], "lrr")


def sensitivity_table(cfg) -> None:
    t = pd.read_csv(path(cfg, "outputs") / "sensitivity.csv")
    w = t.pivot(index=["scenario", "forecast_months"], columns="reversion_months", values="reserve_rate_pct")
    names = {"baseline": "Baseline", "severely_adverse": "Severely adverse"}
    rows = []
    for i, ((scen, f), r) in enumerate(w.iterrows()):
        if i and f == w.index.get_level_values(1).min():
            rows.append("\\midrule")
        cells = " & ".join(f"{v:.2f}" for v in r.values)
        rows.append(f"{names[scen]} & {f} months & {cells} \\\\")
    head = " & ".join("No reversion" if c == 0 else f"{c}-month reversion" for c in w.columns)
    _write("sensitivity.tex", f"Scenario & Forecast window & {head}", rows, "ll" + "r" * len(w.columns))


def portfolio_stats(cfg) -> None:
    """Printed, not written: a few numbers quoted in the text."""
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    p = load_panel(path(cfg, "processed"))
    s = p[(p["period"] == as_of) & (p["event"] == CURRENT) & (p["upb"] > 0)]
    s = s.dropna(subset=["rate", "rem_term", "orig_value", "hpi_orig"])   # same filter as the ECL step
    w = s["upb"] / s["upb"].sum()
    print(f"snapshot loans {len(s):,}, upb {s['upb'].sum() / 1e9:.2f}bn, avg upb {s['upb'].mean():,.0f}")
    for c in ["fico", "dti", "cltv", "rate", "rem_term", "age", "incentive"]:
        print(f"  balance-weighted {c}: {(w * s[c]).sum():.2f}")
    print(f"  share with incentive < -1: {100 * (w * (s['incentive'] < -1)).sum():.1f}%")
    print(f"  vintage shares: {(s.groupby('vintage')['upb'].sum() / s['upb'].sum() * 100).round(1).to_dict()}")


def make_all(cfg) -> None:
    for f in [sample_table, pd_table, lgd_table, ecl_table, backtest_table, sensitivity_table]:
        f(cfg)
