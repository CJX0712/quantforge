SHELL := /bin/bash
PY ?= python
OUT ?= results

.DEFAULT_GOAL := help
.PHONY: help install dev lint fmt test cov demo verify clean docker-build lock

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## install the package
	$(PY) -m pip install .

dev: ## install with development extras
	$(PY) -m pip install -e ".[dev]"

lint: ## run ruff
	$(PY) -m ruff check .

fmt: ## auto-fix lint issues
	$(PY) -m ruff check . --fix

test: ## run the test suite
	$(PY) -m pytest tests -q

cov: ## run tests with a coverage report
	$(PY) -m pytest tests -q --cov=quantforge --cov-report=term-missing

demo: ## run the benchmark and write $(OUT)/benchmark.json
	$(PY) -m quantforge demo --outdir $(OUT)

verify: ## re-run and check the determinism guarantee
	$(PY) -m quantforge verify --outdir $(OUT)

info: ## print runtime capabilities
	$(PY) -m quantforge info

lock: ## regenerate requirements.lock.txt from the current environment
	$(PY) -m pip freeze > requirements.lock.txt

docker-build: ## build the container image
	docker build -t quantforge:0.1.0 .

clean: ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +