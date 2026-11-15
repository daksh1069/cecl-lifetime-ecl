"""Turn raw Freddie Mac files into the modelling tables.

panel     one row per loan-month while the loan is at risk, with the outcome
          of that month: 0 = still paying, 1 = default, 2 = prepaid.
defaults  one row per defaulted loan with its realised loss, for the LGD table.
"""
import numpy as np
import pandas as pd

from .features import current_ltv
from .freddie import find_files, read_orig, read_perf

CURRENT, DEFAULT, PREPAY, CENSORED = 0, 1, 2, 3


def build_vintage(orig: pd.DataFrame, perf: pd.DataFrame, feats: pd.DataFrame, cfg: dict, year: int):
    """feats is the output of macro.macro_features (lagged drivers by month);
    year is the origination year of the Freddie Mac file."""
    rule = cfg["default"]
    perf = perf.sort_values(["loan_id", "period"]).reset_index(drop=True)

    seriously_late = perf["dq"] >= rule["dpd_status"]
    if rule.get("exclude_forbearance"):
        # months in a forbearance plan (mostly COVID-19) are not counted as default
        seriously_late &= ~perf["forbearance"]
    is_default = seriously_late | perf["zb_code"].isin(rule["zb_codes"])
    is_prepay = perf["zb_code"].eq("01") & ~is_default
    is_other_exit = perf["zb_code"].notna() & ~is_default & ~is_prepay   # repurchase, note sale ...
    perf["event"] = np.select([is_default, is_prepay, is_other_exit], [DEFAULT, PREPAY, CENSORED], CURRENT)

    # Balance at the start of the month (the zero-balance row itself reports 0)
    orig_upb = perf["loan_id"].map(orig.set_index("loan_id")["orig_upb"])
    perf["upb_lag"] = perf.groupby("loan_id")["upb"].shift(1).fillna(orig_upb)

    # Keep each loan up to and including its first event
    had_event = (perf["event"] > 0).astype(int)
    earlier_events = had_event.groupby(perf["loan_id"]).cumsum() - had_event
    panel = perf[(earlier_events == 0) & (perf["event"] != CENSORED)].copy()

    loans = orig.copy()
    loans["orig_period"] = loans["first_payment"] - pd.offsets.MonthBegin(1)
    loans["orig_value"] = loans["orig_upb"] / (loans["oltv"] / 100)
    loans["hpi_orig"] = loans["orig_period"].map(feats["hpi"])
    panel = panel.merge(
        loans[["loan_id", "fico", "dti", "oltv", "orig_rate", "orig_value", "orig_period", "hpi_orig"]],
        on="loan_id", how="inner",
    ).merge(feats, left_on="period", right_index=True, how="inner")

    panel["rate"] = panel["rate"].fillna(panel["orig_rate"])
    panel["cltv"] = current_ltv(panel["upb_lag"], panel["orig_value"], panel["hpi"], panel["hpi_orig"])
    panel["incentive"] = panel["rate"] - panel["mort30"]
    panel["vintage"] = year
    keep = ["loan_id", "period", "event", "age", "upb", "upb_lag", "rate", "rem_term",
            "fico", "dti", "oltv", "orig_value", "hpi_orig", "cltv", "incentive",
            "unrate", "hpi_yoy", "vintage"]
    panel = panel[keep]

    # Defaulted loans and how they ended
    d = panel.loc[panel["event"] == DEFAULT, ["loan_id", "period", "upb_lag", "cltv", "vintage"]]
    d = d.rename(columns={"period": "default_period", "upb_lag": "default_upb", "cltv": "default_cltv"})
    last = perf.groupby("loan_id").agg(final_zb=("zb_code", "last"), actual_loss=("actual_loss", "sum"))
    d = d.merge(last, left_on="loan_id", right_index=True)
    liquidated = d["final_zb"].isin(rule["zb_codes"])
    cured = d["final_zb"].eq("01")
    # Actual loss is reported as a positive number for a loss (removal UPB + accrued
    # interest + expenses - sale proceeds - recoveries; checked against the raw fields).
    # Loans that paid off after defaulting count as zero loss; loans still open are
    # unresolved (NaN) and left out of the LGD table.
    d["loss"] = np.select([liquidated, cured], [d["actual_loss"].fillna(0.0), 0.0], np.nan)
    d["severity"] = d["loss"] / d["default_upb"]
    return panel, d


def build_all(cfg: dict, feats: pd.DataFrame, raw_dir, out_dir) -> None:
    rng = np.random.default_rng(cfg["data"]["seed"])
    n_keep = cfg["data"]["loans_per_vintage"]
    panel_dir = out_dir / "panel"
    panel_dir.mkdir(exist_ok=True)
    defaults, found = [], 0
    for year in range(cfg["data"]["vintage_start"], cfg["data"]["vintage_end"] + 1):
        orig_files, perf_files = find_files(raw_dir, year)
        if not orig_files or not perf_files:
            print(f"{year}: no files found, skipped")
            continue
        found += 1
        orig = read_orig(orig_files)
        if n_keep and len(orig) > n_keep:
            orig = orig.iloc[np.sort(rng.choice(len(orig), n_keep, replace=False))]
        perf = read_perf(perf_files, loan_ids=set(orig["loan_id"]))
        panel, d = build_vintage(orig, perf, feats, cfg, year)
        panel.to_parquet(panel_dir / f"{year}.parquet", index=False)
        defaults.append(d)
        counts = panel["event"].value_counts()
        print(f"{year}: {len(orig):,} loans, {len(panel):,} loan-months, "
              f"{counts.get(DEFAULT, 0):,} defaults, {counts.get(PREPAY, 0):,} prepayments")
    if not found:
        raise FileNotFoundError(f"no Freddie Mac files under {raw_dir}; see README, section Data")
    pd.concat(defaults, ignore_index=True).to_parquet(out_dir / "defaults.parquet", index=False)


def load_panel(out_dir, columns=None) -> pd.DataFrame:
    files = sorted((out_dir / "panel").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no panel in {out_dir}; run `python -m cecl build` first")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
