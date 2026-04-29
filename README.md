*Este proyecto ha sido creado como parte del currículo de 42 por `<vvan-ach>`.*

## Descripción

**CallMeMaybe** convierte peticiones en lenguaje natural en llamadas a función estructuradas (JSON) con `prompt`, `fn_name` y `args`, sin responder directamente a la pregunta. El objetivo del proyecto 42 es dominar la **decodificación restringida** : enmascarar los logits token a token para imponer un JSON sintácticamente válido y un esquema acorde a `function_definitions.json`, con un modelo pequeño (Qwen3-0.6B vía `llm_sdk`).

## Instrucciones

### Requisitos

- Python 3.10 o superior
- [uv](https://github.com/astral-sh/uv)

### Instalación

En la raíz del repositorio (incluye `llm_sdk` como dependencia local):

```bash
uv sync --all-extras
```

O con el Makefile:

```bash
make install
```

Si falta `uv.lock` o cambias dependencias, ejecuta `uv lock` antes de `uv sync`.

### Ejecución

Por defecto se leen `data/input/` y se escribe `data/output/function_calling_results.json` (véase `python -m src --help`).

```bash
uv run python -m src
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
make run
make run ARGS='--input data/input/function_calling_tests.json -v'
```

### Depuración y calidad

```bash
make debug
make lint
make lint-strict
```

## Explicación del algoritmo (decodificación restringida)

1. Se construye un contexto con un preámbulo (instrucciones + lista de funciones) y el texto del usuario.
2. En cada paso, el modelo produce logits para el siguiente token.
3. Según el estado de la cadena generada (`START`, texto dentro de `"prompt"`, puente JSON, `fn_name`, `args`, etc.), se ponen a \(-\infty\) los logits de los tokens que romperían la estructura o el esquema.
4. Se elige el token restante de mayor logit (greedy).
5. Para **`fn_name`**, el sujeto exige que la función se elija **solo con el LLM** : se calcula la log-verosimilitud media del nombre completo de cada candidata bajo el contexto actual y se inyecta el ganador; el texto JSON usa exactamente ese identificador (evitando el error típico BPE «suma de decode por token ≠ decode conjunto»).
6. Si el máscara deja cero tokens válidos, un paso de recuperación avanza con fragmentos permitidos por la gramática (y el mismo objetivo LLM para completar `fn_name`).

## Decisiones de diseño

- **Pydantic** para los esquemas de entrada/salida.
- **Numpy** solo para manipular logits (softmax estable).
- **Sin** PyTorch / Transformers / heurísticas de enrutamiento (BM25, palabras clave) en `src/` : el enrutamiento de función cumple el sujeto vía **solo LM**.
- Caché de cadenas decodificadas por id de token para acelerar el enmascarado.
- Rechazo del resultado si no valida el esquema; entonces se emite un objeto de respaldo tipado para no romper el pipeline (la precisión depende sobre todo del modelo bajo restricciones).

## Análisis de rendimiento

- Objetivo del sujeto: JSON válido casi siempre, precisión de función/argumentos > ~95 %, tiempo total de pruebas razonable.
- El cuello de botella es el coste por paso (vocabulario × forward del modelo) y la longitud de los argumentos tipo cadena.

## Retos

- Alinear texto JSON y secuencia de tokens con tokenizadores BPE.
- Mantener el masque correcto en `args` (números, cadenas con comillas/escapes).
- Cumplir la regla del sujeto: **elección de función únicamente con el LLM**.

## Estrategia de pruebas

- Ejecutar `make run` sobre `data/input/function_calling_tests.json`.
- Comprobar `data/output/function_calling_results.json` (JSON válido, claves `prompt` / `fn_name` / `args`, tipos según definiciones).
- Casos límite recomendados por el sujeto: cadenas vacías, números grandes, caracteres especiales, prompts ambiguos, funciones con muchos parámetros.

## Ejemplos de uso

```bash
uv sync --all-extras
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
make run ARGS='-v --log-every 32 --input data/input/function_calling_tests.json'
```

## Recursos

- [Pydantic](https://docs.pydantic.dev/)
- [NumPy](https://numpy.org/doc/)
- Documentación del modelo [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Idea de **constrained decoding** / guided generation en LLMs

### Uso de IA

La IA se utilizó de forma puntual para revisar texto (README), sugerir correcciones de bugs (alineación tokenizer / JSON) y comprobar coherencia con el enunciado. Toda modificación de código fue revisada manualmente; la lógica de decodificación restringida y el cumplimiento del sujeto (elección de función solo por LM) son responsabilidad del autor.
