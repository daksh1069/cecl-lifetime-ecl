# Lifetime Expected Credit Loss Under CECL

A competing-risk model for a residential mortgage portfolio.
FIM 590 (004), Risk Management in Commercial Banks, Fall 2026.

Team: Abhishek Tandon, Daksh Kumar, Shreyansh Sharma.

The code builds the CECL allowance for a portfolio of Freddie Mac mortgages from
start to finish: a monthly multinomial logit for default and prepayment (PD), a
loss-severity table by current LTV (LGD), the amortisation schedule (EAD), and an
engine that sums discounted PD x LGD x EAD over each loan's remaining life under
the Fed's baseline and severely adverse scenarios.

## Setup

Needs Python 3.11 and `make`.

```bash
make setup                  # creates .venv and installs the package
source .venv/bin/activate
make test                   # unit tests
make synthetic              # full pipeline on made-up loans, about 30 seconds
```

With the data in place, `make all` runs the real pipeline in about a minute.

`make synthetic` needs no downloads beyond FRED (`make macro`). It writes to
`data/synthetic/` and `outputs_synthetic/`. Those numbers are meaningless and
must not appear in the report.

## Data

Nothing under `data/` is committed. Each teammate downloads their own copy.

| # | What | Where it goes | Status |
|---|------|---------------|--------|
| # | What | Where it goes | How |
|---|------|---------------|-----|
| 1 | Freddie Mac Single-Family Loan-Level Dataset (SFLLD), **sample** files 2005–2025 | `data/raw/freddie/` | manual download |
| 2 | Unemployment, FHFA house price index, 30-year mortgage rate | `data/raw/macro/` | `make macro` |
| 3 | Fed 2026 scenarios: baseline, severely adverse, historic (domestic CSVs) | `data/raw/scenarios/` | manual download |
| 4 | 2026 DFAST results (first-lien mortgage loss rate, 1.5%) | already in `config/default.yaml` | — |

**1. Freddie Mac loans.** Register (free) on the Freddie Mac SFLLD download
page and accept the licence, which does not allow redistributing the data. Not
the CRT/Clarity disclosure page, which covers risk-transfer deals only. Download
from the *Sample File* column (`sample_2005.zip` ... `sample_2025.zip`) and unzip
anywhere below `data/raw/freddie/`. Each holds `sample_orig_YYYY.txt`
(origination: FICO, LTV, DTI, note rate, balance) and `sample_perf_YYYY.txt`
(monthly performance, with realised losses). The sample has 50,000 loans per
year; the config keeps a random 5,000 per year (`data.loans_per_vintage`). The
release used here has performance through June 2026, about 6.4 GB unzipped.

**2. Macro history.** `make macro` pulls three FRED series, no API key needed:
`UNRATE`, `HPIPONM226S` (FHFA purchase-only HPI, monthly) and `MORTGAGE30US`.

**3. Scenarios.** From the Fed's "Dodd-Frank Act Stress Tests 2026" page, under
Scenario Data, download the **Final** versions of *Supervisory Baseline
Domestic*, *Supervisory Severely Adverse Domestic* and *Historic Domestic*
(CSV). Keep the downloaded names; `config/default.yaml` points at them. The
historic file supplies the 2025 Q4 jump-off quarter.

**4. DFAST benchmark.** *2026 Federal Reserve Stress Test Results* (PDF),
Table 9: domestic first-lien mortgages lose $22.5bn, 1.5% of average balances
over 2026 Q1–2028 Q1 under the severely adverse scenario.

## Running the real pipeline

```bash
make macro        # FRED history
make build        # raw files -> data/processed/panel/*.parquet, defaults.parquet
make fit          # PD coefficients and LGD table
make ecl          # reserve under each scenario, DFAST comparison
make backtest     # out-of-time PD test (train through 2019, test after)
make sensitivity  # reserve vs forecast window and reversion period
make all          # build through sensitivity in one go
```

Result tables land in `outputs/` as CSV. Every setting (vintages, sample size,
default definition, reserve date, forecast and reversion windows) is in
`config/default.yaml`.

## Layout

```
config/            default.yaml, synthetic.yaml (overrides for the smoke test)
src/cecl/
  freddie.py       readers for the raw Freddie Mac files
  macro.py         FRED download, scenario loader, forecast -> reversion -> long-run path
  panel.py         loan-month panel with outcomes; defaults table with realised losses
  features.py      design matrix, shared by fitting and projection
  models.py        competing-risk logit, LGD table, amortisation
  engine.py        lifetime ECL
  synthetic.py     made-up data for the smoke test
  __main__.py      pipeline steps (python -m cecl <step>)
tests/             unit tests
notebooks/         exploration and figures
report/            report.tex -> final submission PDF (make report)
outputs/           result tables (not committed)
```

## Modelling choices

- **Default** is the first month a loan is 180+ days past due, or leaves through
  a short sale, third-party sale or REO disposition. Months in a forbearance
  plan (mostly COVID-19) do not count toward the 180-day rule. A loan is
  followed until its first default or prepayment; repurchases and note sales
  are censored.
- **PD features:** FICO, DTI, mark-to-market LTV (balance over original value
  moved with the national HPI), loan age, refinance incentive (note rate minus
  market rate), unemployment, year-on-year HPI growth. Macro inputs are national
  and lagged one month.
- **LGD** is realised loss over balance at default, by current-LTV bucket.
  Defaulted loans that later paid off count as zero loss; loans still unresolved
  are left out.
- **Refinance incentive** (note rate minus market rate) is floored at -0.5 points
  in the PD model. Below that, prepayment is housing turnover and stops falling;
  without the floor, prepayment in 2023-25 was under-predicted by more than half.
- **EAD** is the scheduled balance at the start of each month. Prepayment enters
  through the survival probability.
- **Reversion** is applied to the macro inputs: after the forecast window,
  unemployment, the mortgage rate and HPI growth move linearly to their averages
  since 2000. The proposal described reverting to long-run *loss rates*; reverting
  the inputs gives one consistent path and no jump in house prices. Say so in the
  report, or change `macro.build_path` if the team prefers output reversion.
- Loans already 180+ days past due at the reserve date are outside the snapshot.

## Not built yet

- Results sections of `report/report.tex`.
- Gradient boosting PD model with SHAP: dropped to keep the model simple.

`make figures` writes the charts used in the slides and report to `report/figures/`.

## Deadlines

- Nov 18: class presentation, 10 to 15 minutes including Q&A.
- Dec 2, 11:59 PM: final submission. `make submission` builds `submission.zip`
  with the report PDF, code and result tables. One upload for the group.
