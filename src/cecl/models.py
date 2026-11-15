"""The three model components: competing-risk PD, LGD lookup, EAD amortisation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .features import FEATURES, design


class CompetingRiskLogit:
    """Monthly multinomial logit: 0 = keep paying (base), 1 = default, 2 = prepay."""

    def __init__(self, params=None, columns=None):
        self.params = None if params is None else np.asarray(params, float)   # (n_features, 2)
        self.columns = columns                                                # indices into FEATURES
        self.table = None

    def fit(self, panel: pd.DataFrame) -> "CompetingRiskLogit":
        X = design(panel)
        y = panel["event"].to_numpy()
        ok = np.isfinite(X).all(axis=1)
        X, y = X[ok], y[ok]
        # a feature with no variation in the sample cannot be estimated
        self.columns = [0] + [j for j in range(1, X.shape[1]) if X[:, j].std() > 0]
        names = [FEATURES[j] for j in self.columns]
        res = sm.MNLogit(y, pd.DataFrame(X[:, self.columns], columns=names)).fit(maxiter=200, disp=0)
        self.params = res.params.to_numpy()
        self.table = pd.concat(
            {"coef": res.params, "se": res.bse, "p": res.pvalues}, axis=1
        ).rename(columns={0: "default", 1: "prepay"}, level=1)
        self.n_obs, self.llf = int(res.nobs), float(res.llf)
        return self

    def predict(self, d) -> np.ndarray:
        """Probabilities (n, 3) in the order current / default / prepay."""
        eta = design(d)[:, self.columns] @ self.params
        e = np.exp(np.column_stack([np.zeros(len(eta)), eta]))
        return e / e.sum(axis=1, keepdims=True)

    def save(self, file: Path) -> None:
        file.write_text(json.dumps({"params": self.params.tolist(), "columns": self.columns}))

    @classmethod
    def load(cls, file: Path) -> "CompetingRiskLogit":
        return cls(**json.loads(file.read_text()))


class LGDTable:
    """Loss severity by current-LTV bucket, weighted by balance at default."""

    def __init__(self, bins, values=None):
        self.bins = list(bins)
        self.values = None if values is None else np.asarray(values, float)

    def fit(self, defaults: pd.DataFrame, min_obs: int = 30) -> "LGDTable":
        d = defaults.dropna(subset=["loss", "default_cltv"])
        d = d[d["default_upb"] > 0]
        if d.empty:
            raise ValueError("no resolved defaults to fit the LGD table")
        if d["loss"].sum() <= 0:
            raise ValueError("total realised loss is not positive: check the sign of actual_loss "
                             "in the raw files (positive = loss in the 2026 release)")
        overall = d["loss"].sum() / d["default_upb"].sum()
        bucket = np.digitize(d["default_cltv"], self.bins)
        g = d.groupby(bucket).agg(n=("loss", "size"), loss=("loss", "sum"), upb=("default_upb", "sum"))
        g = g.reindex(range(len(self.bins) + 1))
        sev = (g["loss"] / g["upb"]).where(g["n"] >= min_obs, overall)
        self.values = sev.clip(0, 1).to_numpy()
        lo = [0] + self.bins
        hi = self.bins + [np.inf]
        self.table = pd.DataFrame({"cltv_from": lo, "cltv_to": hi, "n": g["n"].fillna(0).astype(int).values,
                                   "severity": self.values})
        return self

    def predict(self, cltv) -> np.ndarray:
        return self.values[np.digitize(np.nan_to_num(cltv, nan=70.0), self.bins)]

    def save(self, file: Path) -> None:
        file.write_text(json.dumps({"bins": self.bins, "values": self.values.tolist()}))

    @classmethod
    def load(cls, file: Path) -> "LGDTable":
        return cls(**json.loads(file.read_text()))


def monthly_payment(balance, annual_rate_pct, months):
    """Level payment that amortises `balance` over `months` at the note rate."""
    r = np.asarray(annual_rate_pct, float) / 1200
    n = np.maximum(np.asarray(months, float), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = balance * r / (1 - (1 + r) ** -n)
    return np.where(r > 0, pay, balance / n)
