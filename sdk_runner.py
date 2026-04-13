import json
import os
import sys
from typing import Any
from llm_sdk import Small_LLM_Model
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


_model: Small_LLM_Model | None = None


def get_model() -> Small_LLM_Model:
    global _model
    if _model is None:
        sdk_device = os.getenv("SDK_DEVICE")
        if sdk_device:
            _model = Small_LLM_Model(device=sdk_device)
        else:
            _model = Small_LLM_Model()
    return _model


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        method = payload.get("method")
        model = get_model()

        if method == "encode":
            text = str(payload.get("text", ""))
            encoded = model.encode(text)
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()[0]
            result: Any = encoded
        elif method == "decode":
            token_ids = payload.get("token_ids", [])
            result = model.decode(token_ids)
        elif method == "get_logits_from_input_ids":
            input_ids = payload.get("input_ids", [])
            result = model.get_logits_from_input_ids(input_ids)
        elif method == "get_path_to_vocabulary_json":
            if hasattr(model, "get_path_to_vocabulary_json"):
                result = model.get_path_to_vocabulary_json()
            else:
                result = model.get_path_to_tokenizer_file()
        elif method == "get_path_to_tokenizer_file":
            result = model.get_path_to_tokenizer_file()
        else:
            raise ValueError(f"Unsupported method: {method}")

        print(json.dumps({"result": result}))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
