"""Made-up loans and scenarios in the real file formats, to smoke-test the
pipeline before the Freddie Mac download is available. The numbers mean nothing."""
import numpy as np
import pandas as pd

from .config import ROOT
from .freddie import ORIG_COLS, PERF_COLS
from .macro import load_macro
from .models import monthly_payment


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def write_synthetic(cfg: dict, loans_per_vintage: int = 1500) -> None:
    raw = ROOT / cfg["paths"]["freddie_raw"]
    scen_dir = ROOT / cfg["paths"]["scenarios_raw"]
    if "synthetic" not in str(raw) or "synthetic" not in str(scen_dir):
        raise SystemExit("refusing to write synthetic data into the real data folders; "
                         "run with --config config/synthetic.yaml")
    raw.mkdir(parents=True, exist_ok=True)
    scen_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["data"]["seed"])
    macro = load_macro(cfg)
    end = pd.Timestamp(cfg["ecl"]["as_of"])

    for year in range(cfg["data"]["vintage_start"], cfg["data"]["vintage_end"] + 1):
        orig_rows, perf_parts = [], []
        for i in range(loans_per_vintage):
            loan_id = f"S{year % 100:02d}Q1{i:07d}"
            first = pd.Timestamp(year, int(rng.integers(1, 13)), 1)
            months = pd.date_range(first, end, freq="MS")
            n = len(months)
            fico = int(np.clip(rng.normal(745, 45), 550, 840))
            ltv = int(np.clip(rng.normal(76, 14), 30, 97))
            dti = int(np.clip(rng.normal(34, 9), 10, 50))
            upb0 = float(round(rng.lognormal(12.3, 0.45), -3))
            rate = round(float(macro["mort30"].asof(first - pd.DateOffset(months=2)) + rng.normal(0, 0.25)), 3)
            orig_rows.append({"fico": fico, "first_payment": f"{first:%Y%m}", "dti": dti, "orig_upb": upb0,
                              "oltv": ltv, "ocltv": ltv, "orig_rate": rate, "orig_term": 360,
                              "loan_id": loan_id, "state": "NC", "occupancy": "P", "purpose": "P",
                              "amort_type": "FRM"})

            # scheduled balance at the end of month t (t = 0 is the first payment month)
            r, pay = rate / 1200, monthly_payment(upb0, rate, 360)
            t = np.arange(1, n + 1)
            bal = np.maximum(upb0 * (1 + r) ** t - pay * ((1 + r) ** t - 1) / r, 0)
            m = macro.reindex(months).ffill()
            cltv = ltv * bal / upb0 / (m["hpi"].values / macro["hpi"].asof(first))
            h_default = _sigmoid(-7.6 - 0.012 * (fico - 745) + 0.04 * (cltv - 75) + 0.3 * (m["unrate"].values - 5))
            h_prepay = _sigmoid(-4.6 + 0.9 * np.clip(rate - m["mort30"].values, -1, 3))
            u = rng.random((n, 2))
            t_def = int(np.argmax(u[:, 0] < h_default)) if (u[:, 0] < h_default).any() else n
            t_pre = int(np.argmax(u[:, 1] < h_prepay)) if (u[:, 1] < h_prepay).any() else n
            t_def, t_pre = max(t_def, 1), max(t_pre, 1)

            upb, dq = bal.copy(), np.zeros(n, int)
            zb = np.full(n, "", dtype=object)
            loss = np.full(n, "", dtype=object)
            zb_upb = np.full(n, "", dtype=object)
            last = n
            if t_pre < t_def:                       # voluntary prepayment
                zb[t_pre], zb_upb[t_pre], upb[t_pre], last = "01", round(bal[t_pre - 1], 2), 0.0, t_pre + 1
            elif t_def < n:                         # stops paying at t_def, resolved 9 months later
                stop = min(t_def + 9, n)
                dq[t_def:stop] = np.arange(1, stop - t_def + 1)
                upb[t_def:stop] = bal[t_def - 1]
                if t_def + 9 < n:
                    k, b = t_def + 9, bal[t_def - 1]
                    if rng.random() < 0.7:          # liquidated with a loss (reported positive)
                        sev = np.clip(0.15 + 0.005 * (cltv[t_def] - 60) + rng.normal(0, 0.08), 0.02, 0.9)
                        zb[k], loss[k] = "09", round(sev * b, 2)
                    else:                           # cured and paid off
                        zb[k] = "01"
                    zb_upb[k], upb[k], dq[k], last = round(b, 2), 0.0, 0, k + 1
            sl = slice(0, last)
            perf_parts.append(pd.DataFrame({
                "loan_id": loan_id, "period": months[sl].strftime("%Y%m"), "upb": np.round(upb[sl], 2),
                "dq_status": dq[sl].astype(str), "age": np.arange(n)[sl], "rem_term": (359 - np.arange(n))[sl],
                "zb_code": zb[sl], "rate": rate, "actual_loss": loss[sl], "zb_upb": zb_upb[sl],
            }))
        pd.DataFrame(orig_rows).reindex(columns=ORIG_COLS).to_csv(
            raw / f"sample_orig_{year}.txt", sep="|", header=False, index=False)
        pd.concat(perf_parts).reindex(columns=PERF_COLS).to_csv(
            raw / f"sample_perf_{year}.txt", sep="|", header=False, index=False)
        print(f"{year}: wrote {loans_per_vintage} synthetic loans")

    # Two scenarios shaped like the Fed's domestic CSVs: jump-off quarter + 13 projected quarters
    quarters = pd.period_range(end.to_period("Q"), periods=14, freq="Q")
    q = np.arange(14)
    u0, m0 = macro["unrate"].asof(end), macro["mort30"].asof(end)
    paths = {
        "baseline": (u0 + 0.02 * q, m0 - 0.03 * q, 100 * 1.008 ** q),
        "severely_adverse": (u0 + (10 - u0) * np.minimum(q / 6, 1) - 0.3 * np.maximum(q - 6, 0),
                             m0 - 0.15 * np.minimum(q, 6),
                             100 * (1 - 0.28 * np.minimum(q / 8, 1)) * 1.01 ** np.maximum(q - 8, 0)),
    }
    for name, (unrate, mort, hpi) in paths.items():
        pd.DataFrame({
            "Scenario Name": name, "Date": [f"{p.year} Q{p.quarter}" for p in quarters],
            "Unemployment rate": np.round(unrate, 2), "Mortgage rate": np.round(mort, 2),
            "House Price Index (Level)": np.round(hpi, 2),
        }).to_csv(scen_dir / cfg["ecl"]["scenarios"][name], index=False)
    print(f"wrote synthetic scenarios to {scen_dir}")
