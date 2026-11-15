PY := .venv/bin/python
SYN := --config config/synthetic.yaml

.PHONY: setup macro build fit ecl backtest sensitivity figures all synthetic test report submission

setup:            ## create the venv and install the package
	python3.11 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

macro:            ## download FRED history
	$(PY) -m cecl macro

build fit ecl backtest sensitivity figures:   ## individual pipeline steps on the real data
	$(PY) -m cecl $@

all:              ## build -> fit -> ecl -> backtest -> sensitivity on the real data
	$(PY) -m cecl all

synthetic:        ## end-to-end smoke test on made-up loans (results in outputs_synthetic/)
	$(PY) -m cecl synthetic $(SYN)
	$(PY) -m cecl all $(SYN)

test:
	$(PY) -m pytest -q

report:           ## build report/report.pdf
	cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex

submission: report   ## zip code + report + result tables for Moodle (no raw data)
	rm -f submission.zip
	zip -r submission.zip README.md pyproject.toml Makefile config src tests notebooks \
		report/report.pdf report/report.tex report/figures report/tables outputs \
		-x "*/__pycache__/*" "*.egg-info/*" "*/.gitkeep" "*/.DS_Store"
