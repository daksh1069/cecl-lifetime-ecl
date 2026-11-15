"""Macro history from FRED, Fed stress test scenarios, and the forward path
(forecast window -> reversion -> long-run) that the ECL engine consumes."""
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import path

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DRIVERS = ["unrate", "hpi", "mort30"]


def fetch_fred(cfg: dict) -> None:
    """Download each configured FRED series to paths.macro_raw/<name>.csv."""
    out = path(cfg, "macro_raw")
    for name, sid in cfg["macro"]["fred"].items():
        resp = requests.get(FRED_URL, params={"id": sid}, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = ["date", name]
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{sid}: {len(df)} rows, {df['date'].iloc[0]} to {df['date'].iloc[-1]}")


def load_macro(cfg: dict) -> pd.DataFrame:
    """Monthly history indexed by month start, columns unrate / hpi / mort30."""
    raw = path(cfg, "macro_raw")
    cols = {}
    for name in cfg["macro"]["fred"]:
        f = raw / f"{name}.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing; run `python -m cecl macro` first")
        s = pd.read_csv(f, parse_dates=["date"]).set_index("date")[name]
        cols[name] = pd.to_numeric(s, errors="coerce").resample("MS").mean()
    # Interior gaps are filled linearly. The October 2025 unemployment rate was never
    # published (federal government shutdown); dropping the month instead would drop
    # every loan-month of November 2025 and every loan originated in October 2025.
    return pd.DataFrame(cols).interpolate(limit_area="inside").dropna()


def long_run(macro: pd.DataFrame, cfg: dict) -> dict:
    """Long-run averages the forecast reverts to (HPI as mean monthly log growth)."""
    m = macro.loc[cfg["macro"]["long_run_start"]:]
    return {
        "unrate": float(m["unrate"].mean()),
        "mort30": float(m["mort30"].mean()),
        "hpi_growth": float(np.log(m["hpi"]).diff().mean()),
    }


def load_scenario(file: Path, history: Path | None = None) -> pd.DataFrame:
    """Read a Fed supervisory scenario CSV (domestic variables, quarterly).

    Columns are matched by name fragment so the files can be used as downloaded:
    a date column like "2026 Q1", plus the unemployment rate, mortgage rate and
    house price index. Returns a frame indexed by the last month of each quarter.
    `history` is the Fed's historic domestic CSV; its quarters before the first
    scenario quarter are put in front, which supplies the jump-off row.
    """
    out = _read_fed_csv(file)
    if history is not None:
        hist = _read_fed_csv(history)
        out = pd.concat([hist[hist.index < out.index[0]], out])
    return out


def _read_fed_csv(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    lower = {c.lower(): c for c in df.columns}

    def find(fragment: str) -> str:
        hits = [orig for low, orig in lower.items() if fragment in low]
        if not hits:
            raise KeyError(f"{file.name}: no column containing '{fragment}' in {list(df.columns)}")
        return hits[0]

    def quarter_end(text: str) -> pd.Timestamp:
        year = re.search(r"(\d{4})", str(text))
        qtr = re.search(r"[Qq]\s*([1-4])", str(text))
        if not (year and qtr):
            raise ValueError(f"{file.name}: cannot read quarter from '{text}'")
        return pd.Timestamp(int(year.group(1)), int(qtr.group(1)) * 3, 1)

    out = pd.DataFrame({
        "unrate": pd.to_numeric(df[find("unemployment")], errors="coerce").values,
        "mort30": pd.to_numeric(df[find("mortgage rate")], errors="coerce").values,
        "hpi": pd.to_numeric(df[find("house price")], errors="coerce").values,
    }, index=[quarter_end(v) for v in df[find("date")]])
    return out.sort_index().dropna()


def reversion_weights(horizon: int, forecast_months: int, reversion_months: int) -> np.ndarray:
    """Weight on the scenario in projection months 1..horizon: 1 inside the forecast
    window, falling linearly to 0 over the reversion window, 0 afterwards."""
    k = np.arange(1, horizon + 1)
    if reversion_months == 0:
        return (k <= forecast_months).astype(float)
    return np.clip(1 - (k - forecast_months) / reversion_months, 0, 1)


def build_path(macro: pd.DataFrame, scenario: pd.DataFrame, as_of: pd.Timestamp,
               forecast_months: int, reversion_months: int, horizon: int, lr: dict) -> pd.DataFrame:
    """History up to as_of followed by `horizon` projected months.

    Projection: scenario values inside the forecast window, then each driver
    reverts linearly to its long-run average (the HPI in growth-rate terms, so
    the level never jumps), then stays there.
    """
    hist = macro.loc[:as_of]
    if hist.index[-1] != as_of:
        raise ValueError(f"macro history ends {hist.index[-1]:%Y-%m}, before as_of {as_of:%Y-%m}")
    future = pd.date_range(as_of + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    # Scenario HPI is on its own scale: rescale so it joins the FHFA level at as_of.
    before = scenario.loc[:as_of]
    if len(before):
        anchor = before["hpi"].iloc[-1]
    else:
        anchor = scenario["hpi"].iloc[0]
        warnings.warn("scenario has no row at or before as_of; assuming flat house prices "
                      "up to its first quarter. Add the last historical quarter to the CSV.")
    scen = scenario.loc[scenario.index > as_of].copy()
    scen["hpi"] = hist["hpi"].iloc[-1] * scen["hpi"] / anchor
    if len(scen) == 0 or scen.index[-1] < future[min(forecast_months, horizon) - 1]:
        warnings.warn("scenario is shorter than the forecast window; last values are held flat")

    # Quarterly -> monthly, starting from the actuals at as_of
    grid = pd.concat([hist.iloc[[-1]], pd.DataFrame(index=future, columns=DRIVERS, dtype=float)])
    grid.update(scen)
    grid["hpi"] = np.log(grid["hpi"])
    grid = grid.interpolate(method="linear").ffill()

    w = reversion_weights(horizon, forecast_months, reversion_months)
    proj = pd.DataFrame(index=future)
    for col in ["unrate", "mort30"]:
        proj[col] = w * grid[col].values[1:] + (1 - w) * lr[col]
    growth = w * np.diff(grid["hpi"].values) + (1 - w) * lr["hpi_growth"]
    proj["hpi"] = hist["hpi"].iloc[-1] * np.exp(np.cumsum(growth))
    return pd.concat([hist, proj[DRIVERS]])


def macro_features(macro: pd.DataFrame, lag: int) -> pd.DataFrame:
    """Model inputs by month: drivers plus year-on-year HPI growth (%), lagged so
    the row for month t only uses data published by t - lag."""
    out = macro[DRIVERS].copy()
    out["hpi_yoy"] = 100 * (out["hpi"] / out["hpi"].shift(12) - 1)
    return out.shift(lag)
