PY ?= python

.PHONY: check verify test demo mini full seeds

check:
	$(PY) -m src.validation

verify:
	$(PY) -m src.validation --run-tests

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m experiments.04_final --config configs/demo.yaml

mini:
	$(PY) -m experiments.04_final --config configs/mini.yaml

full:
	$(PY) -m experiments.04_final --config configs/full.yaml

seeds:
	$(PY) -m experiments.05_seeds --config configs/full.yaml --n-seeds 10
