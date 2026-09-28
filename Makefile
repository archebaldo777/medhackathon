# Python 3.12 или 3.13: make install PYTHON=python3.12
PYTHON ?= python3

.PHONY: install train benchmark test demo serve docker-build docker-run

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

train:
	.venv/bin/python scripts/train.py

benchmark:
	.venv/bin/python scripts/benchmark.py

test:
	.venv/bin/pytest

demo:
	.venv/bin/python -m src analyze "dataset/for tests" --output outputs/demo --zip

serve:
	.venv/bin/python -m src serve

docker-build:
	./scripts/build.sh

docker-run:
	./scripts/run.sh
