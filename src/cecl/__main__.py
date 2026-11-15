"""Pipeline steps. Usage: python -m cecl <step> [--config config/synthetic.yaml]

macro        download unemployment, HPI and mortgage rate history from FRED
build        raw Freddie Mac files -> loan-month panel + defaults table
fit          fit the PD model and LGD table on all data up to the reserve date
ecl          lifetime ECL under each scenario, plus the DFAST-style comparison
backtest     out-of-time test of the PD model
sensitivity  reserve vs forecast window and reversion period
figures      charts and LaTeX tables for the report (report/figures/, report/tables/)
synthetic    write made-up loans and scenarios for a smoke test (synthetic config only)
all          build, fit, ecl, backtest, sensitivity
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import macro as mc
from .config import load_config, path
from .engine import lifetime_ecl, summarise
from .models import CompetingRiskLogit, LGDTable
from .panel import CURRENT, DEFAULT, PREPAY, build_all, load_panel


def step_macro(cfg):
    mc.fetch_fred(cfg)


def step_build(cfg):
    feats = mc.macro_features(mc.load_macro(cfg), cfg["macro"]["lag_months"])
    build_all(cfg, feats, path(cfg, "freddie_raw"), path(cfg, "processed"))


def step_fit(cfg):
    proc, out = path(cfg, "processed"), path(cfg, "outputs")
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    panel = load_panel(proc)
    model = CompetingRiskLogit().fit(panel[panel["period"] <= as_of])
    model.save(proc / "pd_model.json")
    model.table.to_csv(out / "pd_coefficients.csv")
    print(model.table.round(4).to_string())

    defaults = pd.read_parquet(proc / "defaults.parquet")
    lgd = LGDTable(cfg["lgd"]["cltv_bins"]).fit(defaults[defaults["default_period"] <= as_of],
                                                cfg["lgd"]["min_obs"])
    lgd.save(proc / "lgd_table.json")
    lgd.table.to_csv(out / "lgd_table.csv", index=False)
    print(lgd.table.round(4).to_string(index=False))


def _ecl_inputs(cfg):
    proc = path(cfg, "processed")
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    panel = load_panel(proc)
    snapshot = panel[(panel["period"] == as_of) & (panel["event"] == CURRENT) & (panel["upb"] > 0)]
    snapshot = snapshot.dropna(subset=["rate", "rem_term", "orig_value", "hpi_orig"])
    if snapshot.empty:
        raise ValueError(f"no open loans at {as_of:%Y-%m}; check ecl.as_of against the data")
    macro = mc.load_macro(cfg)
    scen_dir = path(cfg, "scenarios_raw")
    history = cfg["ecl"].get("scenario_history")
    scenarios = {name: mc.load_scenario(scen_dir / file, scen_dir / history if history else None)
                 for name, file in cfg["ecl"]["scenarios"].items()}
    return (as_of, snapshot, macro, scenarios,
            CompetingRiskLogit.load(proc / "pd_model.json"), LGDTable.load(proc / "lgd_table.json"))


def _run(cfg, as_of, snapshot, macro, scenario, model, lgd, forecast_months, reversion_months):
    horizon = int(snapshot["rem_term"].max())
    macro_path = mc.build_path(macro, scenario, as_of, forecast_months, reversion_months,
                               horizon, mc.long_run(macro, cfg))
    feats = mc.macro_features(macro_path, cfg["macro"]["lag_months"])
    ecl, monthly = lifetime_ecl(snapshot, feats, as_of, model, lgd, cfg["ecl"]["discount"])
    return summarise(snapshot, ecl, monthly), monthly, macro_path.loc[as_of:]


def step_ecl(cfg):
    out = path(cfg, "outputs")
    as_of, snapshot, macro, scenarios, model, lgd = _ecl_inputs(cfg)
    f, r = cfg["ecl"]["forecast_months"], cfg["ecl"]["reversion_months"]
    rows = {}
    for name, scen in scenarios.items():
        rows[name], monthly, macro_path = _run(cfg, as_of, snapshot, macro, scen, model, lgd, f, r)
        monthly.to_csv(out / f"ecl_monthly_{name}.csv")
        macro_path.to_csv(out / f"macro_path_{name}.csv")
    summary = pd.DataFrame(rows).T
    summary.to_csv(out / "ecl_summary.csv")
    print(summary.round(3).to_string())

    # DFAST comparison: follow the severely adverse scenario for its full 9 quarters
    # (no reversion inside the window) and compare the cumulative loss rate.
    if "severely_adverse" in scenarios:
        s, _, _ = _run(cfg, as_of, snapshot, macro, scenarios["severely_adverse"], model, lgd, 27, 12)
        dfast = pd.Series({"model_9q_loss_rate_pct": s["loss_rate_9q_pct"],
                           "published_9q_loss_rate_pct": cfg["dfast"]["published_first_lien_loss_rate"]})
        dfast.to_csv(out / "dfast_comparison.csv", header=["value"])
        print(dfast.to_string())


def step_backtest(cfg):
    out = path(cfg, "outputs")
    as_of = pd.Timestamp(cfg["ecl"]["as_of"])
    cut = pd.Timestamp(cfg["pd"]["train_end"])
    panel = load_panel(path(cfg, "processed"))
    panel = panel[panel["period"] <= as_of]
    train, test = panel[panel["period"] <= cut], panel[panel["period"] > cut].copy()
    model = CompetingRiskLogit().fit(train)
    p = model.predict(test)
    ok = np.isfinite(p).all(axis=1)
    test, p = test[ok], p[ok]
    test["p_default"], test["p_prepay"] = p[:, 1], p[:, 2]
    metrics = pd.Series({
        "train_end": f"{cut:%Y-%m}",
        "train_rows": len(train),
        "test_rows": len(test),
        "auc_default": roc_auc_score(test["event"] == DEFAULT, test["p_default"]),
        "auc_prepay": roc_auc_score(test["event"] == PREPAY, test["p_prepay"]),
        "actual_default_rate_pct": 100 * (test["event"] == DEFAULT).mean(),
        "predicted_default_rate_pct": 100 * test["p_default"].mean(),
        "actual_prepay_rate_pct": 100 * (test["event"] == PREPAY).mean(),
        "predicted_prepay_rate_pct": 100 * test["p_prepay"].mean(),
    })
    metrics.to_csv(out / "backtest_metrics.csv", header=["value"])
    print(metrics.to_string())
    by_month = test.assign(default=test["event"] == DEFAULT, prepay=test["event"] == PREPAY) \
        .groupby("period")[["default", "p_default", "prepay", "p_prepay"]].mean()
    by_month.to_csv(out / "backtest_by_month.csv")


def step_sensitivity(cfg):
    as_of, snapshot, macro, scenarios, model, lgd = _ecl_inputs(cfg)
    grid = cfg["sensitivity"]
    rows = []
    for name, f, r in itertools.product(scenarios, grid["forecast_months"], grid["reversion_months"]):
        s, _, _ = _run(cfg, as_of, snapshot, macro, scenarios[name], model, lgd, f, r)
        rows.append({"scenario": name, "forecast_months": f, "reversion_months": r,
                     "reserve_rate_pct": s["reserve_rate_pct"], "ecl": s["ecl"]})
    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "outputs") / "sensitivity.csv", index=False)
    print(table.pivot(index=["scenario", "forecast_months"], columns="reversion_months",
                      values="reserve_rate_pct").round(3).to_string())


def step_figures(cfg):
    from . import figures, report_tables
    figures.make_all(cfg)
    report_tables.make_all(cfg)


def step_synthetic(cfg):
    from .synthetic import write_synthetic
    write_synthetic(cfg)


STEPS = {"macro": step_macro, "build": step_build, "fit": step_fit, "ecl": step_ecl,
         "backtest": step_backtest, "sensitivity": step_sensitivity, "figures": step_figures,
         "synthetic": step_synthetic}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=[*STEPS, "all"])
    parser.add_argument("--config", help="YAML file whose keys override config/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    names = ["build", "fit", "ecl", "backtest", "sensitivity"] if args.step == "all" else [args.step]
    for name in names:
        print(f"\n== {name} ==")
        STEPS[name](cfg)


if __name__ == "__main__":
    main()
