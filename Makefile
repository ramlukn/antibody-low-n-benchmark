PY ?= .venv/bin/python

.PHONY: help setup data embed baselines run figures all clean-results

help:
	@echo "make setup      create .venv and install requirements"
	@echo "make data       download TDC datasets, report splits and leakage"
	@echo "make embed      cache frozen ESM-2 embeddings (~4 min on an M-series Mac)"
	@echo "make baselines  Phase 2 audit of the cheap descriptor baselines"
	@echo "make run        the full learning-curve sweep (~40 min on 12 cores)"
	@echo "make figures    tables and figures from results/learning_curves.csv"
	@echo "make all        everything, in order"

setup:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/01_prepare_data.py

embed:
	$(PY) scripts/02_embed.py

baselines:
	$(PY) scripts/03_baselines.py

run:
	$(PY) scripts/04_learning_curves.py

quick:
	$(PY) scripts/04_learning_curves.py --quick

figures:
	$(PY) scripts/06_figures.py

all: data embed baselines run figures

clean-results:
	rm -f results/*.csv figures/*.png
