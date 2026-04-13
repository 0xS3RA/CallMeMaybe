import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel


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


_FN_KEY = '"fn_name": "'
_MAX_STEPS = 512


def _type_ok(val: Any, t: str) -> bool:
    if t == "string":
        return isinstance(val, str)
    if t == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if t == "boolean":
        return isinstance(val, bool)
    return True


def _fallback(prompt: str, fns: List[FunctionDefinition]) -> OutputItem:
    if not fns:
        return OutputItem(prompt=prompt, fn_name="fn_unknown", args={})
    fd = fns[0]
    dflt = {"string": "", "number": 0.0, "boolean": False}
    args = {k: dflt.get(p.type, None) for k, p in fd.parameters.items()}
    return OutputItem(prompt=prompt, fn_name=fd.name, args=args)


def _valid(item: OutputItem, fns: List[FunctionDefinition]) -> bool:
    m = {f.name: f for f in fns}
    fd = m.get(item.fn_name)
    if not fd or set(item.args) != set(fd.parameters):
        return False
    return all(
        _type_ok(item.args[k], fd.parameters[k].type) for k in fd.parameters
    )


def _load_vocab(qwen: Any) -> Dict[str, int]:
    with open(qwen.get_path_to_vocab_file(), encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("vocab JSON invalide")
    return {
        k: v for k,
        v in raw.items() if isinstance(k, str)
        and isinstance(v, int)
    }


def _masked_logits(
    logits: np.ndarray, vocab: Dict[str, int], keep: Callable[[str], bool]
) -> np.ndarray:
    out = np.full_like(logits, -np.inf, dtype=np.float64)
    n = len(logits)
    for piece, tid in vocab.items():
        if 0 <= tid < n and piece and keep(piece):
            out[tid] = logits[tid]
    return out


def _logits(qwen: Any, prefix: str, gen: str) -> np.ndarray:
    ids = qwen.encode(prefix + gen).tolist()[0]
    return np.asarray(qwen.get_logits_from_input_ids(ids), dtype=np.float64)


def _preamble(fns: List[FunctionDefinition]) -> str:
    lines = [
        "Route the user to one function. Reply with one JSON object only.",
        'Shape: {"prompt":"<user message>","fn_name":"<id>","args":{...}}',
        "fn_name must be one of the ids below;",
        " args must include every parameter with correct JSON types.",
        "Functions:",
    ]
    for fn in fns:
        ps = ", ".join(f"{k} ({v.type})" for k, v in fn.parameters.items())
        lines.append(f"  - {fn.name}: {fn.description} — {ps}")
    lines.append("Match the main action; fill args from the user text.")
    return "\n".join(lines)


def _stream_write(stream: bool, s: str) -> None:
    if stream:
        print(s, end="", flush=True)


def generate_one(
    prompt: str,
    qwen: Any,
    fns: List[FunctionDefinition],
    vocab: Dict[str, int],
    *,
    verbose: bool = False,
    label: str = "",
    stream: bool = True,
) -> OutputItem:
    prefix = (
            _preamble(fns) + "\n\nUser:\n" + prompt
            + "\n\nValid JSON (complete the fragment):\n"
    )
    esc = json.dumps(prompt, ensure_ascii=False)[1:-1]
    gen = '{"prompt": "' + esc + '", "fn_name": "'
    names_q = [f.name + '"' for f in fns]
    by_name = {f.name: f for f in fns}
    tag = f"[{label}] " if label else ""

    _stream_write(stream, gen)

    for _ in range(_MAX_STEPS):
        tail = gen.split(_FN_KEY, 1)[1] if _FN_KEY in gen else ""
        log = _logits(qwen, prefix, gen)

        def keep_fn_name(piece: str) -> bool:
            return any(n.startswith(tail + piece) for n in names_q)

        masked = _masked_logits(log, vocab, keep_fn_name)
        if not np.any(np.isfinite(masked)):
            if verbose:
                print(f"{tag}masque fn_name vide", file=sys.stderr, flush=True)
            break
        piece = qwen.decode([int(np.argmax(masked))])
        gen += piece
        _stream_write(stream, piece)
        if piece.endswith('"'):
            break

    raw = gen.split(_FN_KEY, 1)[1] if _FN_KEY in gen else ""
    if not raw.endswith('"'):
        _stream_write(stream, "\n")
        return _fallback(prompt, fns)
    chosen = raw[:-1]
    fd = by_name.get(chosen)
    if fd is None:
        _stream_write(stream, "\n")
        return _fallback(prompt, fns)

    bridge = ', "args": {'
    gen += bridge
    _stream_write(stream, bridge)
    params = list(fd.parameters.items())

    for i, (pname, pinfo) in enumerate(params):
        keypart = f'"{pname}": '
        gen += keypart
        _stream_write(stream, keypart)
        start = len(gen)

        if pinfo.type == "string":
            gen += '"'
            _stream_write(stream, '"')
            for _ in range(_MAX_STEPS):
                piece = qwen.decode(
                    [int(np.argmax(_logits(qwen, prefix, gen)))]
                )
                prev = len(gen)
                gen += piece
                if '"' in piece:
                    gen = gen[: gen.rfind('"') + 1]
                _stream_write(stream, gen[prev:])
                if '"' in piece:
                    break
            else:
                _stream_write(stream, "\n")
                return _fallback(prompt, fns)

        elif pinfo.type == "number":
            allowed = set("0123456789.,-+eE")
            for _ in range(_MAX_STEPS):
                frag = gen[start:]
                core = frag.strip().replace("Ġ", "").replace("▁", "").strip()
                short = len(core) > 24

                def keep_num(p: str) -> bool:
                    if not p or "\n" in p or "\r" in p:
                        return False
                    if not set(p).issubset({",", "}"} if short else allowed):
                        return False
                    return not ("." in core and "." in p)

                log = _logits(qwen, prefix, gen)
                m = _masked_logits(log, vocab, keep_num)
                tid = int(np.argmax(m if np.any(np.isfinite(m)) else log))
                piece = qwen.decode([tid])
                prev = len(gen)
                gen += piece
                if "," in piece or "}" in piece:
                    gen = gen.rstrip(",} \n\t")
                _stream_write(stream, gen[prev:])
                if "," in piece or "}" in piece:
                    break
            else:
                _stream_write(stream, "\n")
                return _fallback(prompt, fns)

        elif pinfo.type == "boolean":
            for _ in range(_MAX_STEPS):
                frag = gen[start:]
                log = _logits(qwen, prefix, gen)

                def keep_bool(piece: str) -> bool:
                    return any(
                        x.startswith(frag + piece) for x in ("true", "false")
                    )

                m = _masked_logits(log, vocab, keep_bool)
                tid = int(np.argmax(m if np.any(np.isfinite(m)) else log))
                piece = qwen.decode([tid])
                gen += piece
                _stream_write(stream, piece)
                if gen[start:].strip() in ("true", "false"):
                    break
            else:
                _stream_write(stream, "\n")
                return _fallback(prompt, fns)
        else:
            _stream_write(stream, "\n")
            return _fallback(prompt, fns)

        if i < len(params) - 1:
            sep = ", "
            gen += sep
            _stream_write(stream, sep)

    closing = "}}"
    gen += closing
    _stream_write(stream, closing + "\n")

    try:
        item = OutputItem(**json.loads(gen.strip()))
        if _valid(item, fns):
            return item
    except Exception:
        if os.environ.get("CALLMEMYBE_DEBUG"):
            print("fail:", repr(gen[-200:]), file=sys.stderr, flush=True)

    return _fallback(prompt, fns)


def _load_json_list(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} doit être un tableau JSON")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="CallMeMaybe")
    ap.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    ap.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    ap.add_argument(
        "--functions", default="data/input/function_definitions.json"
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true"
    )
    ap.add_argument(
        "--no-stream",
        action="store_true",
        help="Désactive l'affichage en direct du JSON sur stdout.",
    )
    args = ap.parse_args()

    fn_path = args.functions
    default_path = "data/input/functions_definition.json"
    if not os.path.exists(fn_path):
        if os.path.exists(default_path):
            fn_path = default_path
        else:
            raise SystemExit(f"Error: file not found: {fn_path}")

    try:
        t0 = time.perf_counter()
        if args.verbose:
            print("Chargement du modèle…", file=sys.stderr, flush=True)
        qwen = Small_LLM_Model()
        if args.verbose:
            print(
                f"Modèle prêt en {time.perf_counter() - t0:.1f}s",
                file=sys.stderr, flush=True
            )

        prompts_raw = _load_json_list(args.input)
        prompts = [
            x if isinstance(x, str) else x["prompt"]
            for x in prompts_raw
            if isinstance(x, str) or (
                isinstance(x, dict) and isinstance(x.get("prompt"), str)
            )
        ]
        functions = [FunctionDefinition(**x) for x in _load_json_list(fn_path)]
        vocab = _load_vocab(qwen)
        if args.verbose:
            print(
                f"{len(prompts)} prompts, "
                f"{len(functions)} fonctions, "
                f"{len(vocab)} tokens vocab",
                file=sys.stderr,
                flush=True,
            )
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
        raise SystemExit(f"Error: {e}") from e

    out: List[OutputItem] = []
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        lab = f"{i + 1}/{n}"
        if args.verbose:
            p = (
                prompt[:72].replace("\n", " ")
                + ("…" if len(prompt) > 72 else "")
            )
            print(f"[{lab}] {p!r}", file=sys.stderr, flush=True)
        t1 = time.perf_counter()
        try:
            if not args.no_stream:
                print(f"\n--- {lab} ---", flush=True)
            out.append(
                generate_one(
                    prompt,
                    qwen,
                    functions,
                    vocab,
                    verbose=args.verbose,
                    label=lab,
                    stream=not args.no_stream,
                )
            )
        except Exception:
            out.append(_fallback(prompt, functions))
        if args.verbose:
            print(
                f"[{lab}] {time.perf_counter() - t1:.1f}s → {out[-1].fn_name}",
                file=sys.stderr,
                flush=True,
            )

    d = os.path.dirname(args.output)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            [x.model_dump() for x in out],
            f,
            ensure_ascii=False, indent=2
        )

    if args.verbose:
        print("Terminé.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
