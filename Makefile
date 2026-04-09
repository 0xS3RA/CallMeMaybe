 SDK_VENV := .venv-sdk
PY_VERSION := 3.12
SDK_PY := $(SDK_VENV)/bin/python
PYTHON := uv run python

SGOINFRE_PATH := /goinfre/vvan-ach/CallMeMaybe_cache
export UV_CACHE_DIR := $(SGOINFRE_PATH)/uv
export HF_HOME      := $(SGOINFRE_PATH)/hf
export PYTHONPYCACHEPREFIX := $(SGOINFRE_PATH)/pycache

.PHONY: install install-sdk run debug clean lint lint-strict

prep-cache:
	@mkdir -p $(UV_CACHE_DIR) $(HF_HOME) $(PYTHONPYCACHEPREFIX)

install: prep-cache
	uv sync --python $(PY_VERSION) --all-extras
	$(MAKE) install-sdk

install-sdk: prep-cache
	@if [ ! -d "$(SDK_VENV)" ]; then uv venv "$(SDK_VENV)" --python $(PY_VERSION); fi
	UV_PROJECT_ENVIRONMENT=$(abspath $(SDK_VENV)) uv sync --project llm_sdk
	uv pip install --python "$(SDK_PY)" "numpy>=1.26.4" "pydantic>=2.12.5"

run: prep-cache
	$(SDK_PY) -m src

debug: prep-cache
	$(SDK_PY) -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__
	# rm -rf $(SGOINFRE_PATH)

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict 
