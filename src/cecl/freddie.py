"""Readers for the Freddie Mac Single-Family Loan-Level Dataset (SFLLD).

Files are pipe-delimited with no header. Layout as checked against the release
with performance through June 2026: 31 origination fields; 34 performance
fields plus a trailing pipe. If Freddie Mac changes the layout, ORIG_COLS and
PERF_COLS are the only things to update.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pcomp
import pyarrow.csv as pcsv

ORIG_COLS = [
    "fico", "first_payment", "first_time_buyer", "maturity", "msa", "mi_pct",
    "units", "occupancy", "ocltv", "dti", "orig_upb", "oltv", "orig_rate",
    "channel", "ppm", "amort_type", "state", "property_type", "zip3", "loan_id",
    "purpose", "orig_term", "n_borrowers", "seller", "super_conforming",
    "pre_harp_id", "program", "harp", "valuation_method", "io", "unused_31",
]

PERF_COLS = [
    "loan_id", "period", "upb", "dq_status", "age", "rem_term", "defect_date",
    "mod_flag", "zb_code", "zb_date", "rate", "deferred_upb", "ddlpi",
    "mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "expenses",
    "legal_costs", "maintenance_costs", "taxes_insurance", "misc_expenses",
    "actual_loss", "mod_cost", "step_mod", "deferral", "eltv", "zb_upb",
    "dq_accrued_interest", "dq_disaster", "assistance_code", "month_mod_cost",
    "interest_bearing_upb", "unused_33", "servicer", "_trailing",
]

ORIG_KEEP = ["loan_id", "fico", "first_payment", "dti", "orig_upb", "oltv",
             "orig_rate", "orig_term", "state", "occupancy", "purpose"]
PERF_KEEP = ["loan_id", "period", "upb", "dq_status", "age", "rem_term",
             "zb_code", "rate", "actual_loss", "zb_upb", "assistance_code"]


def find_files(raw_dir: Path, year: int) -> tuple[list[Path], list[Path]]:
    """Origination and performance files for one vintage year, anywhere below raw_dir.

    Sample files: sample_orig_YYYY.txt + sample_perf_YYYY.txt (older releases: sample_svcg_YYYY.txt).
    Full standard files: historical_data_YYYYQn.txt + historical_data_time_YYYYQn.txt.
    The sample files win if both are present.
    """
    orig = sorted(raw_dir.rglob(f"sample_orig_{year}.txt"))
    perf = sorted(raw_dir.rglob(f"sample_perf_{year}.txt")) or sorted(raw_dir.rglob(f"sample_svcg_{year}.txt"))
    if not orig:
        orig = sorted(raw_dir.rglob(f"historical_data_{year}Q[1-4].txt"))
        perf = sorted(raw_dir.rglob(f"historical_data_time_{year}Q[1-4].txt"))
    return orig, perf


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_orig(files: list[Path]) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(f, sep="|", header=None, names=ORIG_COLS, usecols=ORIG_KEEP,
                     dtype=str, keep_default_na=False, na_values=[""]) for f in files],
        ignore_index=True,
    )
    for col in ["fico", "dti", "orig_upb", "oltv", "orig_rate", "orig_term"]:
        df[col] = _num(df[col])
    # Freddie Mac codes for "not available"
    df["fico"] = df["fico"].where(df["fico"].between(300, 850))
    df["dti"] = df["dti"].where(df["dti"] < 999)
    df["oltv"] = df["oltv"].where(df["oltv"] < 999)
    df["first_payment"] = pd.to_datetime(df["first_payment"], format="%Y%m")
    return df


def read_perf(files: list[Path], loan_ids=None) -> pd.DataFrame:
    """Monthly performance rows, optionally only for `loan_ids`. Uses pyarrow,
    which reads a 450 MB file in a few seconds."""
    parts = []
    for f in files:
        table = pcsv.read_csv(
            f,
            read_options=pcsv.ReadOptions(column_names=PERF_COLS, block_size=64 << 20),
            parse_options=pcsv.ParseOptions(delimiter="|", quote_char=False),
            convert_options=pcsv.ConvertOptions(column_types={c: pa.string() for c in PERF_COLS},
                                                include_columns=PERF_KEEP),
        )
        if loan_ids is not None:
            table = table.filter(pcomp.is_in(table["loan_id"], value_set=pa.array(list(loan_ids))))
        parts.append(table.to_pandas())
    df = pd.concat(parts, ignore_index=True).replace("", np.nan)
    for col in ["upb", "age", "rem_term", "rate", "actual_loss", "zb_upb"]:
        df[col] = _num(df[col])
    df["period"] = pd.to_datetime(df["period"], format="%Y%m")

    # Delinquency status: months past due; "RA" (REO acquisition) -> 99
    status = df["dq_status"].fillna("0").str.strip()
    df["dq"] = np.where(status.isin(["RA", "R"]), 99, _num(status).fillna(0)).astype(int)

    # Zero balance code as a two-character string ("01", "09", ...), NaN while active
    zb = _num(df["zb_code"])
    df["zb_code"] = zb.map(lambda v: f"{int(v):02d}", na_action="ignore").astype(object)
    df["forbearance"] = df["assistance_code"].eq("F")
    return df.drop(columns=["dq_status", "assistance_code"])
