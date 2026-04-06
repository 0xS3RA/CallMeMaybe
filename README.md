*Este proyecto ha sido creado como parte del currículo de 42 por <login1>.*

## Description

`CallMeMaybe` is a function-calling project that maps natural-language prompts to structured JSON function calls.
The goal is to ensure valid, schema-conformant JSON output through constrained decoding with a small language model.

## Instructions

### Prerequisites

- Python 3.10+
- `uv`

### Installation

Installs both environments:

- Root project dependencies from `pyproject.toml`.
- SDK dependencies from `llm_sdk/pyproject.toml` into `.venv-sdk/`.

```bash
make install
```

Requires network access the first time (PyPI downloads).

### Run

```bash
make run
```

The Makefile sets `SDK_PYTHON` to `.venv-sdk/bin/python` so the SDK runner uses the correct interpreter.

To run without Make (same behavior as `make run`):

```bash
SDK_PYTHON="$(pwd)/.venv-sdk/bin/python" uv run python -m src [--input <input_file>] [--output <output_file>]
```

### Debug

```bash
make debug
```

### Lint

```bash
make lint
```

Optional strict mode:

```bash
make lint-strict
```

## Algorithm Explanation

The implementation follows a constrained token-by-token decoding strategy:

1. Encode the prompt to input IDs.
2. Query the model for logits of the next token.
3. Mask invalid tokens (`-inf`) according to the current JSON-generation state.
4. Select the best remaining token.
5. Repeat until a complete JSON object is produced.

This enforces structure and helps guarantee recoverable JSON output.

## Design Decisions

- Use `pydantic` models for input schema validation.
- Separate data loading from generation flow.
- Keep output schema fixed with keys: `prompt`, `fn_name`, `args`.
- Prefer explicit error messages over silent failures.

## Performance Analysis

Expected targets:

- Near-perfect function and argument selection (>95%).
- 100% parseable JSON output.
- Full test set processing in under 5 minutes on expected hardware.

## Challenges Encountered

- Small models are fragile with unconstrained generation.
- Tokenization boundaries can make strict JSON formatting difficult.
- Schema-safe argument generation requires careful token filtering.

## Testing Strategy

- Validate JSON input parsing failure paths.
- Test missing files and malformed files.
- Run representative prompts for each available function.
- Verify final JSON schema and argument types against function definitions.

## Usage Examples

```bash
uv run python -m src
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Resources

- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [Mypy documentation](https://mypy.readthedocs.io/en/stable/)
- [Flake8 documentation](https://flake8.pycqa.org/en/latest/)

AI usage in this project:

- Assisted in requirement analysis and compliance checklisting.
- Assisted in documentation structuring and command standardization.
- All generated suggestions must be reviewed and understood before submission.
