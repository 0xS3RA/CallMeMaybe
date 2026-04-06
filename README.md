*This project was created as part of the 42 curriculum by <login1>.*

## Description

`CallMeMaybe` takes natural-language prompts and generates JSON function calls.
The goal is not to answer the question directly, but to produce a machine-usable structure with:

- `prompt`
- `fn_name`
- `args`

The core idea is token-by-token generation with invalid-token filtering to keep JSON output stable.

## Instructions

### Prerequisites

- Python 3.10+
- `uv`

### Installation

The project keeps two dependency definitions (`pyproject.toml` at root and `llm_sdk/pyproject.toml`). `make install` builds a single runtime venv at `.venv-sdk` with:

- SDK dependencies from `llm_sdk/pyproject.toml`,
- runtime project dependencies needed by `src` (`numpy`, `pydantic`).

```bash
make install
```

### Run

```bash
make run
```

Equivalent direct command:

```bash
.venv-sdk/bin/python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

### Debug

```bash
make debug
```

### Lint

```bash
make lint
make lint-strict
```

## Algorithm Explanation

At each iteration:

1. compute logits for the next token,
2. determine the current JSON state (`prompt`, `fn_name`, `args`),
3. mask tokens that would break the expected structure,
4. pick the best remaining token.

This mechanism greatly reduces invalid outputs and enforces a consistent JSON shape.

## Design Decisions

- `pydantic` is used to validate input/output structures.
- File parsing (`prompts`, `functions`) is separated from generation.
- Final output is validated against function signatures (name, required args, types).
- If generation is invalid, a schema-compliant fallback is produced to avoid broken JSON.

## Performance Notes

Project targets:

- always-parseable JSON output,
- correct function/argument matching in most cases,
- full run completed in a few minutes on a standard test dataset.

## Challenges

- Small models drift quickly without strict constraints.
- Vocabulary/tokenizer formats vary across SDK implementations.
- I/O errors (missing files, invalid JSON) must be handled cleanly without crashes.

## Testing Strategy

- test simple, ambiguous, and noisy prompts,
- test input errors (missing file, invalid JSON),
- verify every output object matches the expected schema.

## Usage Examples

```bash
make run
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Resources

- [Pydantic](https://docs.pydantic.dev/)
- [NumPy](https://numpy.org/doc/)
- [Mypy](https://mypy.readthedocs.io/en/stable/)
- [Flake8](https://flake8.pycqa.org/en/latest/)

## AI Usage

AI was used to:

- rephrase parts of the documentation,
- sanity-check installation and run steps,
- do quick compliance reviews.

Every change was reviewed and adjusted manually before validation.
