"""Requirement-to-code traceability via embeddings.

Cost reality (Cost Addendum 2.1): 500k tokens on text-embedding-3-small is
~$0.01. This is the cheap answer to 'which code realizes requirement R-12' -
the LLM-everything design pays 100x for the same map.

`embed` is injected so tests and local runs can substitute a deterministic
fake; production wiring uses OpenAI text-embedding-3-small ($0.02/MTok).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from vmodel_contracts import RequirementSet


@dataclass(frozen=True)
class CodeSymbol:
    """One addressable unit of the submitted build."""
    symbol_id: str     # e.g. 'src/routes/bookings.ts:createBooking'
    text: str          # signature + docstring + nearby comments


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def traceability_map(
    requirements: RequirementSet,
    symbols: list[CodeSymbol],
    embed,
    threshold: float = 0.55,
    top_k: int = 3,
) -> dict[str, list[str]]:
    """requirement_id -> ranked symbol ids. Orphans (best score below
    threshold) map to [] - the report flags them as MISSING_CAPABILITY."""
    req_texts = [f"{r.requirement_id} {r.title}: {r.statement}" for r in requirements.requirements]
    sym_texts = [s.text for s in symbols]
    req_vecs = embed(req_texts)
    sym_vecs = embed(sym_texts) if sym_texts else []
    out: dict[str, list[str]] = {}
    for r, rv in zip(requirements.requirements, req_vecs):
        scored = sorted(
            ((cosine(rv, sv), s.symbol_id) for s, sv in zip(symbols, sym_vecs)),
            reverse=True,
        )
        out[r.requirement_id] = [sid for score, sid in scored[:top_k] if score >= threshold]
    return out


def openai_embed(api_key: str, model: str = "text-embedding-3-small"):
    """Production embedding fn. $0.02/MTok; log usage in delivery-api."""
    import httpx

    from .errors import UserFacingError

    def _embed(texts: list[str]) -> list[list[float]]:
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "input": texts},
                timeout=60,
            )
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise UserFacingError(
                "could not reach api.openai.com for embeddings.",
                "network problem or OpenAI outage.",
                "check connectivity / status.openai.com, then retry - or use ollama_embed for a fully offline loop.",
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UserFacingError(
                    "OpenAI rejected the embeddings request (401).",
                    "the OPENAI_API_KEY passed to openai_embed is missing or invalid.",
                    "export a valid OPENAI_API_KEY and retry.",
                ) from e
            raise
        data = resp.json()
        return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]

    return _embed


_JS_SYMBOL = re.compile(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(")


def ollama_embed(model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
    """Local embeddings via Ollama: fully offline, $0. Quality is below
    text-embedding-3-small for nuance, fine for dev-loop traceability."""
    import httpx

    from .errors import UserFacingError

    def _embed(texts: list[str]) -> list[list[float]]:
        try:
            resp = httpx.post(f"{base_url}/api/embed", json={"model": model, "input": texts}, timeout=120)
            resp.raise_for_status()
        except httpx.ConnectError as e:
            raise UserFacingError(
                f"could not reach the local Ollama server at {base_url}.",
                "Ollama is not running (or the base_url is wrong).",
                "run `ollama serve` (or start the Ollama app) and retry.",
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise UserFacingError(
                    f"Ollama does not have embedding model '{model}'.",
                    "the model was never pulled on this machine.",
                    f"run `ollama pull {model}` and retry.",
                ) from e
            raise
        return resp.json()["embeddings"]

    return _embed


def extract_symbols_python(source: str, path: str) -> list[CodeSymbol]:
    """AST-based symbol extraction (Python). JS/TS uses the regex fast path
    below until tree-sitter lands - honest skeleton, named extension point."""
    import ast as _ast

    out = []
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return out
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            doc = _ast.get_docstring(node) or ""
            out.append(CodeSymbol(symbol_id=f"{path}:{node.name}", text=f"{node.name} {doc}"))
    return out


def extract_symbols_js(source: str, path: str) -> list[CodeSymbol]:
    return [CodeSymbol(symbol_id=f"{path}:{m.group(1) or m.group(2)}", text=m.group(0))
            for m in _JS_SYMBOL.finditer(source)]
