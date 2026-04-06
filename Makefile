SDK_VENV := .venv-sdk
SDK_PY := $(SDK_VENV)/bin/python
PYTHON := uv run python

.PHONY: install install-sdk run debug clean lint lint-strict

# Installs root deps (dev/lint) + one runtime venv containing both root and SDK deps.
install:
	uv sync --all-extras
	$(MAKE) install-sdk

install-sdk:
	UV_PROJECT_ENVIRONMENT=$(abspath $(SDK_VENV)) uv sync --all-extras
	UV_PROJECT_ENVIRONMENT=$(abspath $(SDK_VENV)) uv sync --project llm_sdk

run:
	$(SDK_PY) -m src

debug:
	$(SDK_PY) -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
