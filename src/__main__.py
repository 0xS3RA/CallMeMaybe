import argparse
import json
import os
from enum import Enum
from typing import Any, Dict, List

import numpy as np
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel


class Status(Enum):
    START = 1
    IN_PROMPT = 2
    AFTER_PROMPT = 3
    IN_FN_NAME = 4
    AFTER_FN_NAME = 5
    IN_ARGS = 6
    COMPLETED = 7

class PromptItem(BaseModel):
    prompt: str

class ParameterInfo(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: ParameterInfo


class OutputItem(BaseModel):
    prompt: str
    fn_name: str
    args: Dict[str, Any]


def _is_valid_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _default_value_for_type(expected_type: str) -> Any:
    if expected_type == "string":
        return ""
    if expected_type == "number":
        return 0.0
    if expected_type == "boolean":
        return False
    return None


def _build_fallback_item(prompt: str, functions: List[FunctionDefinition]) -> OutputItem:
    if not functions:
        return OutputItem(prompt=prompt, fn_name="fn_unknown", args={})
    first_fn = functions[0]
    args: Dict[str, Any] = {}
    for name, info in first_fn.parameters.items():
        args[name] = _default_value_for_type(info.type)
    return OutputItem(prompt=prompt, fn_name=first_fn.name, args=args)


def _validate_output_item(item: OutputItem, functions: List[FunctionDefinition]) -> bool:
    function_map = {fn.name: fn for fn in functions}
    fn_def = function_map.get(item.fn_name)
    if fn_def is None:
        return False
    if set(item.args.keys()) != set(fn_def.parameters.keys()):
        return False
    for arg_name, arg_info in fn_def.parameters.items():
        if not _is_valid_type(item.args[arg_name], arg_info.type):
            return False
    return True


def get_current_state(generated_text: str, target_prompt: str, function_names: List[str]) -> str:
    if not generated_text:
        return "START"

    if generated_text.startswith('{"prompt": "'):
        if not generated_text.endswith(f'"{target_prompt}", "fn_name": "'):
            if generated_text == '{"prompt": "':
                return "IN_PROMPT"
            if generated_text.endswith(f'"{target_prompt}'):
                return "AFTER_PROMPT"
            return "IN_PROMPT"

    if '"fn_name": "' in generated_text and '", "args": {' not in generated_text:
        if generated_text.endswith('"fn_name": "'):
            return "IN_FN_NAME"
        for name in function_names:
            if generated_text.endswith(f'"fn_name": "{name}'):
                return "AFTER_FN_NAME"
        return "IN_FN_NAME"

    if '"args": {' in generated_text:
        if generated_text.endswith('}}'):
            return "COMPLETED"
        return "IN_ARGS"

    return "UNKNOWN"


def get_vocab(model_sdk: Any) -> Dict[int, str]:
    if hasattr(model_sdk, "get_path_to_vocabulary_json"):
        tokenizer_path = model_sdk.get_path_to_vocabulary_json()
    else:
        tokenizer_path = model_sdk.get_path_to_tokenizer_file()
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_vocab: Dict[str, int]
    if isinstance(data, dict) and "model" in data and isinstance(data["model"], dict) and "vocab" in data["model"]:
        raw_vocab = data["model"]["vocab"]
    elif isinstance(data, dict):
        raw_vocab = data
    else:
        raise ValueError("Invalid vocabulary JSON format.")

    return {int(v): str(k) for k, v in raw_vocab.items()}


def get_functions_definitions(path: str) -> List[FunctionDefinition]:
    with open(path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    if not isinstance(raw_json, list):
        raise ValueError("Function definitions file must contain a JSON array.")
    return [FunctionDefinition(**item) for item in raw_json]


def get_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON array.")

    prompts: List[str] = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item and isinstance(item["prompt"], str):
            prompts.append(item["prompt"])
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CallMeMaybe function-calling generator")
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--functions",
        default="data/input/function_definitions.json",
        help="Path to function definitions JSON",
    )
    return parser.parse_args()


def resolve_functions_path(path: str) -> str:
    if os.path.exists(path):
        return path
    compat_path = "data/input/functions_definition.json"
    if path == "data/input/function_definitions.json" and os.path.exists(compat_path):
        return compat_path
    raise FileNotFoundError(path)


def generate_one(prompt: str, qwen: Any, functions: List[FunctionDefinition], vocab: Dict[int, str]) -> OutputItem:
    function_names = [f.name for f in functions]
    current_tokens = qwen.encode(prompt).tolist()[0]
    generated_json = ""

    for _ in range(200):
        logits = qwen.get_logits_from_input_ids(current_tokens)
        state = get_current_state(generated_json, prompt, function_names)
        for token_id in range(len(logits)):
            token_text = vocab.get(token_id, "").replace("Ġ", " ").replace("Ċ", "\n")
            if state == "START":
                if token_text != '{"prompt": "':
                    logits[token_id] = -float("inf")
            elif state == "IN_PROMPT":
                header = '{"prompt": "'
                content_already_written = generated_json[len(header):]
                remaining_prompt = prompt[len(content_already_written):]
                if not remaining_prompt.startswith(token_text):
                    logits[token_id] = -float("inf")
            elif state == "AFTER_PROMPT":
                target = '", "fn_name": "'
                if not target.startswith(token_text) and not token_text.startswith(target):
                    logits[token_id] = -float("inf")
            elif state == "IN_FN_NAME":
                is_valid = False
                for name in function_names:
                    current_fn_part = generated_json.split('"fn_name": "')[-1]
                    if name.startswith(current_fn_part + token_text):
                        is_valid = True
                        break
                if not is_valid:
                    logits[token_id] = -float("inf")
            elif state == "AFTER_FN_NAME":
                target = '", "args": {'
                if not target.startswith(token_text) and not token_text.startswith(target):
                    logits[token_id] = -float("inf")
            elif state == "IN_ARGS":
                try:
                    chosen_fn_name = generated_json.split('"fn_name": "')[1].split('"')[0]
                    current_fn_def = next(f for f in functions if f.name == chosen_fn_name)
                except (IndexError, StopIteration):
                    logits[token_id] = -float("inf")
                    continue
                args_part = generated_json.split('"args": {')[-1]
                if args_part.count(":") >= len(current_fn_def.parameters):
                    if token_text not in ["}", "}}"]:
                        logits[token_id] = -float("inf")

        next_token_id = int(np.argmax(logits))
        current_tokens.append(next_token_id)
        generated_json += qwen.decode([next_token_id])
        if state == "COMPLETED":
            break

    try:
        parsed = json.loads(generated_json)
        validated = OutputItem(**parsed)
        if _validate_output_item(validated, functions):
            return validated
        return _build_fallback_item(prompt, functions)
    except Exception:
        return _build_fallback_item(prompt, functions)


def write_results(path: str, results: List[OutputItem]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    try:
        qwen = Small_LLM_Model()
        prompts = get_prompts(args.input)
        functions = get_functions_definitions(resolve_functions_path(args.functions))
        vocab = get_vocab(qwen)
    except FileNotFoundError as exc:
        raise SystemExit(f"Error: file not found: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Error: invalid JSON file: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Error during initialization: {exc}") from exc

    results: List[OutputItem] = []
    for prompt in prompts:
        try:
            results.append(generate_one(prompt, qwen, functions, vocab))
        except Exception:
            results.append(_build_fallback_item(prompt, functions))

    write_results(args.output, results)

if __name__ == "__main__":
    main()
