PYTHON := uv run python
SDK_VENV := .venv-sdk
SDK_PY := $(SDK_VENV)/bin/python

.PHONY: install install-sdk run debug clean lint lint-strict

# Project deps + separate SDK deps, each with its own pyproject.toml.
install:
	uv sync --all-extras
	$(MAKE) install-sdk

install-sdk:
	UV_PROJECT_ENVIRONMENT=$(abspath $(SDK_VENV)) uv sync --project llm_sdk

run:
	SDK_PYTHON=$(abspath $(SDK_PY)) $(PYTHON) -m src

debug:
	SDK_PYTHON=$(abspath $(SDK_PY)) $(PYTHON) -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
