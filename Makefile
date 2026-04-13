PY_VERSION := 3.12
ARGS ?=

.PHONY: install run debug clean lint lint-strict

install:
	uv sync --python $(PY_VERSION) --all-extras

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	rm -rf src/__pycache__

lint:
	uv run flake8 . --exclude=.venv,.venv-sdk,llm_sdk
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude llm_sdk --follow-imports=silent

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
