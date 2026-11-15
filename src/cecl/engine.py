"""Lifetime expected credit loss: discounted sum of PD x LGD x EAD over every
remaining month of every loan, along one macro path."""
import numpy as np
import pandas as pd

from .features import current_ltv
from .models import monthly_payment


def lifetime_ecl(snapshot: pd.DataFrame, feats: pd.DataFrame, as_of: pd.Timestamp,
                 pd_model, lgd, discount: bool = True):
    """
    snapshot  loans open at as_of: upb, rate, rem_term, age, fico, dti, orig_value, hpi_orig
    feats     macro.macro_features of the projected path, covering as_of + 1 .. as_of + max term
    Returns (per-loan ECL array, month-by-month portfolio DataFrame).
    """
    bal = snapshot["upb"].to_numpy(float)
    rate = snapshot["rate"].to_numpy(float)
    rem = np.maximum(snapshot["rem_term"].to_numpy(float), 1)
    age0 = snapshot["age"].to_numpy(float)
    orig_value = snapshot["orig_value"].to_numpy(float)
    hpi_orig = snapshot["hpi_orig"].to_numpy(float)
    fico, dti = snapshot["fico"].to_numpy(float), snapshot["dti"].to_numpy(float)

    payment = monthly_payment(bal, rate, rem)
    r = rate / 1200
    surv = np.ones(len(bal))          # probability the loan is still open at the start of the month
    ecl = np.zeros(len(bal))
    rows = []
    for k in range(1, int(rem.max()) + 1):
        month = as_of + pd.DateOffset(months=k)
        m = feats.loc[month]
        alive = rem >= k
        cltv = current_ltv(bal, orig_value, m["hpi"], hpi_orig)
        p = pd_model.predict({
            "fico": fico, "dti": dti, "cltv": cltv, "age": age0 + k,
            "incentive": rate - m["mort30"], "unrate": m["unrate"], "hpi_yoy": m["hpi_yoy"],
        })
        p_default, p_prepay = p[:, 1] * alive, p[:, 2] * alive
        loss = surv * p_default * lgd.predict(cltv) * bal        # EAD = balance at start of month
        pv = loss / (1 + r) ** k if discount else loss
        ecl += pv
        rows.append((month, (surv * bal).sum(), (surv * p_default * bal).sum(),
                     (surv * p_prepay * bal).sum(), loss.sum(), pv.sum()))
        surv = surv * (1 - p_default - p_prepay)
        bal = np.maximum(bal * (1 + r) - payment, 0)             # scheduled amortisation
    monthly = pd.DataFrame(rows, columns=["month", "expected_balance", "default_balance",
                                          "prepay_balance", "loss", "loss_pv"]).set_index("month")
    return ecl, monthly


def summarise(snapshot: pd.DataFrame, ecl: np.ndarray, monthly: pd.DataFrame) -> dict:
    upb = snapshot["upb"].sum()
    return {
        "loans": len(snapshot),
        "upb": upb,
        "ecl": ecl.sum(),
        "reserve_rate_pct": 100 * ecl.sum() / upb,
        "lifetime_loss_undiscounted_pct": 100 * monthly["loss"].sum() / upb,
        "loss_rate_9q_pct": 100 * monthly["loss"].iloc[:27].sum() / upb,
        "lifetime_default_pct": 100 * monthly["default_balance"].sum() / upb,
    }
