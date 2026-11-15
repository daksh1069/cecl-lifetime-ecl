"""Design matrix shared by model fitting and the ECL projection.

Both the historical loan-month panel and the engine's projected months go
through `design`, so a feature can never be defined two different ways.
"""
import numpy as np

FEATURES = [
    "const", "fico", "dti", "cltv", "cltv_over_100", "log_age", "age_years",
    "incentive", "incentive_pos", "unrate", "hpi_yoy",
]

# Below this refinance incentive (note rate - market rate, pct pts) prepayment is
# housing turnover, not refinancing, and stops falling. Without the floor the model
# extrapolates linearly: training data (2005-19) has almost no incentives below -2,
# while 2023-25 is full of them, and prepayment was under-predicted by more than half.
INCENTIVE_FLOOR = -0.5


def current_ltv(balance, orig_value, hpi, hpi_orig):
    """Mark-to-market LTV (%): balance over the original value moved with the HPI."""
    return 100 * balance / (orig_value * hpi / hpi_orig)


def design(d) -> np.ndarray:
    """Build the feature matrix from a DataFrame or dict of equal-length arrays with
    fico, dti, cltv, age (months), incentive (note rate - market rate, pct pts),
    unrate and hpi_yoy. Missing borrower fields are set to the centring value."""
    fico = np.nan_to_num((np.asarray(d["fico"], float) - 740) / 50)
    dti = np.nan_to_num((np.asarray(d["dti"], float) - 35) / 10)
    cltv = np.nan_to_num(np.asarray(d["cltv"], float), nan=70.0)
    age = np.maximum(np.asarray(d["age"], float), 1)
    inc = np.asarray(d["incentive"], float)
    ones = np.ones(len(fico))
    return np.column_stack([
        ones,
        fico,
        dti,
        (cltv - 70) / 10,
        np.maximum(cltv - 100, 0) / 10,
        np.log(age),
        age / 12,
        np.maximum(inc, INCENTIVE_FLOOR),
        np.maximum(inc, 0),
        np.asarray(d["unrate"], float) * ones,
        np.asarray(d["hpi_yoy"], float) * ones,
    ])
