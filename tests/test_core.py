import numpy as np
import pandas as pd
import pytest

from cecl import macro as mc
from cecl.engine import lifetime_ecl
from cecl.models import LGDTable, monthly_payment


def test_payment_amortises_to_zero():
    bal, rate, n = 200_000.0, 6.0, 360
    pay = monthly_payment(bal, rate, n)
    assert pay == pytest.approx(1199.10, abs=0.01)
    for _ in range(n):
        bal = bal * (1 + rate / 1200) - pay
    assert bal == pytest.approx(0, abs=1e-4)


def test_reversion_weights():
    w = mc.reversion_weights(40, forecast_months=24, reversion_months=12)
    assert w[:24].tolist() == [1.0] * 24
    assert w[29] == pytest.approx(0.5)      # month 30, halfway through reversion
    assert w[35:].tolist() == [0.0] * 5
    assert mc.reversion_weights(3, 2, 0).tolist() == [1.0, 1.0, 0.0]


def _flat_macro(months):
    idx = pd.date_range("2020-01-01", periods=months, freq="MS")
    return pd.DataFrame({"unrate": 5.0, "hpi": 100.0, "mort30": 6.0}, index=idx)


def test_path_follows_scenario_then_reverts():
    macro = _flat_macro(36)
    as_of = macro.index[-1]                                   # 2022-12
    scen = pd.DataFrame({"unrate": [5.0, 8.0, 8.0], "mort30": [6.0, 6.0, 6.0], "hpi": [200.0, 180.0, 180.0]},
                        index=pd.to_datetime(["2022-12-01", "2023-03-01", "2023-06-01"]))
    lr = {"unrate": 6.0, "mort30": 6.0, "hpi_growth": 0.0}
    p = mc.build_path(macro, scen, as_of, forecast_months=6, reversion_months=6, horizon=24, lr=lr)
    assert p.loc["2023-03-01", "unrate"] == pytest.approx(8.0)
    assert p.loc["2023-03-01", "hpi"] == pytest.approx(90.0)  # rescaled to the historical level
    assert p.loc["2023-09-01", "unrate"] == pytest.approx(7.0)  # halfway back to the long run
    assert p.loc["2024-06-01", "unrate"] == pytest.approx(6.0)
    assert p["hpi"].iloc[-1] == pytest.approx(90.0)           # level is kept, growth reverts


class ConstantHazard:
    def __init__(self, p_default, p_prepay):
        self.p = (1 - p_default - p_prepay, p_default, p_prepay)

    def predict(self, d):
        return np.tile(self.p, (len(d["fico"]), 1))


def test_engine_matches_hand_calculation():
    """Zero-rate loan, constant hazards and severity: ECL has a closed form."""
    n, upb, pdm, ppm, sev = 12, 12_000.0, 0.01, 0.02, 0.4
    snap = pd.DataFrame({"upb": [upb], "rate": [0.0], "rem_term": [n], "age": [10], "fico": [700.0],
                         "dti": [30.0], "orig_value": [20_000.0], "hpi_orig": [100.0]})
    macro = _flat_macro(60)
    as_of = macro.index[30]
    feats = mc.macro_features(macro, lag=1)
    lgd = LGDTable([80], values=[sev, sev])
    ecl, monthly = lifetime_ecl(snap, feats, as_of, ConstantHazard(pdm, ppm), lgd, discount=True)
    expected = sum((1 - pdm - ppm) ** (k - 1) * pdm * sev * (upb - (k - 1) * upb / n) for k in range(1, n + 1))
    assert ecl[0] == pytest.approx(expected)
    assert len(monthly) == n
