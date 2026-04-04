import sys
import os
import json
import numpy as np
from pydantic import BaseModel
from typing import List, Dict
from llm import Small_LLM_Model
from enum import Enum


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

def get_vocab(model_sdk: Small_LLM_Model) -> Dict[int, str]:
    tokenizer_path = model_sdk.get_path_to_tokenizer_file()
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw_vocab = data['model']['vocab']
    vocab = {v: k for k, v in raw_vocab.items()}
    return vocab

def get_functions_definitions(path: str) -> List[FunctionDefinition]:
    with open(path, 'r') as f:
        raw_json = json.load(f)
    definitions = [FunctionDefinition(**item) for item in raw_json]
    return definitions

def get_prompts(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data: List[Dict[str, str]] = json.load(f)
            return  [item["prompt"] for item in data if "prompt" in item]
    except FileNotFoundError:
        print(f"Error: The file {path} doesn't exist")
        return []
    except json.JSONDecodeError:
        print(f"Error: The file {path} is not a valid json file")

def main() -> None:
    qwen = Small_LLM_Model()
    prompts = get_prompts("data/input/function_calling_tests.json")
    functions = get_functions_definitions("data/input/functions_definitions.json")
    vocab: Dict[int, str] = get_vocab(qwen)
    functions_names = [f.name for f in functions]

    for i, p in enumerate(prompts):
        current_tokens = qwen.encode(p).tolist()[0]
        generated_json = ""
        for _ in range(200):
            logits = qwen.get_logits_from_input_ids(current_tokens)
            state = get_current_state(generated_json, p, functions_names)
            for token_id in range(len(logits)):
                token_text = vocab.get(token_id, "").replace('Ġ', ' ').replace('Ċ', '\n')
                if state == "START":
                    if token_text != '{"prompt": "':
                        logits[token_id] = -float("inf")
                elif state == "IN_PROMPT":
                    header = '{"prompt": "'
                    content_already_written = generated_json[len(header):]
                    remaining_prompt = p[len(content_already_written):]
                    if not remaining_prompt.startswith(token_text):
                        logits[token_id] = -float("inf")
                elif state == "AFTER_PROMPT":
                    target = '", "fn_name": "'
                    if not target.startswith(token_text) and not token_text.startswith(target):
                        logits[token_id] = -float("inf")
                elif state == "IN_FN_NAME":
                    is_valid = False
                    for name in functions_names:
                        current_fn_part = generated_json.split('"fn_name": "')[-1]
                        if (name).startswith(current_fn_part + token_text):
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
                    execpt (IndexError, StopIteration):
                        logits[token_id] = -float("inf")
                        continue
                    args_part = generated_json.split('"args: {')[-1]
                    if args_part.count(':') >= len(current_fn_def.parameters):
                        if token_text not in ["}", "}}"]:
                            logits[token_id] = -float("inf")


            next_token_id = int(np.argmax(logits))
            current_tokens.append(next_token_id)
            new_piece = qwen.decode([next_token_id])
            generated_json += new_piece
            if state == "COMPLETED":
                break

if __name__ == "__main__":
    main()
