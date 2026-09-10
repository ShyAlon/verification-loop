"""Model routing for the ONE LLM call in verification: the semantic-residue
judge. Providers: ollama (DEFAULT - local-first, ADR-0018), deepseek/openai/anthropic (opt-in),
a local judge is fine for development; verdicts shown to founders should come
from the strong model, the verdicts ARE the product).
"""
from __future__ import annotations

import json
import os
import re as _re
import sys
import time
from dataclasses import dataclass

_DOTENV: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    """Config files, parsed once per process, most-shared first:

    1. <require-root>/.env - the SHARED config one directory up
       (vmodel-spec-app/scripts/switch-model.sh writes there; one file drives
       every process). Present only in the sibling-repo layout; absent in a
       standalone checkout and nothing changes.
    2. ./.env - repo-local overrides for the rare per-repo deviation.

    Real environment variables always win over both files (see _env), and
    os.environ is never mutated - no dependency, no side effects."""
    global _DOTENV
    if _DOTENV is not None:
        return _DOTENV
    _DOTENV = {}
    from pathlib import Path

    for p in (Path("..") / ".env", Path(".env")):
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            _DOTENV[key.strip()] = value.strip().strip('"').strip("'")
    return _DOTENV


def _env(key: str) -> str | None:
    """One setting lookup: the real environment wins, the repo-local .env
    fills the gap. Every external config read goes through here so a .env
    written by switch-model.sh works for API keys too, not just routing."""
    return os.environ.get(key) or _dotenv().get(key)


# ---------------------------------------------------------------- guardrails
# Every LLM call passes through these limits. A hung ollama generation costs a
# hang at full GPU; the same hang on a paid API is a bill. All four knobs are
# env/.env-overridable; the defaults match "demo on a 16GB laptop, or cents".
_CALL_TIMEOUT_S = 600   # wall-clock budget per call, client retries included
_STALL_TIMEOUT_S = 180  # a stream this quiet is hung, not thinking
_MAX_CALLS = 200        # what ONE process run may spend, in LLM calls
_CALLS_USED = 0


_MAX_TOKENS_ENV = "VMODEL_LLM_MAX_TOKENS"  # per-call generation ceiling override (unset: stage default)


def _max_tokens_limit(stage_default: int) -> int:
    """Per-call generation-length ceiling. Each stage ships its own default
    (StageModel.max_tokens); VMODEL_LLM_MAX_TOKENS overrides it globally - a
    looping generation hits a length ceiling, not only a clock ceiling."""
    raw = _env(_MAX_TOKENS_ENV)
    try:
        v = int(raw) if raw else stage_default
    except ValueError:
        v = stage_default
    return v if v > 0 else stage_default


def _truncation_reason(resp) -> str | None:
    meta = getattr(resp, "response_metadata", None) or {}
    for key in ("done_reason", "finish_reason", "stop_reason"):  # ollama / openai-compat / anthropic
        if meta.get(key) in ("length", "max_tokens"):
            return str(meta[key])
    return None


def _check_truncated(resp, stage: str, cfg) -> None:
    """A generation cut at the token ceiling is a failure, not an answer: name
    it here, or it surfaces downstream as a confusing parse/validation error."""
    if _truncation_reason(resp) is None:
        return
    from .errors import UserFacingError

    ceiling = _max_tokens_limit(cfg.max_tokens)
    _log_llm({"event": "llm_call_error", "provider": cfg.provider, "model": cfg.model, "stage": stage,
              "error": {"kind": "token_ceiling", "ceiling_tokens": ceiling}})
    raise UserFacingError(
        f"stage '{stage}': the {cfg.provider}/{cfg.model} call hit the generation-length "
        f"ceiling ({ceiling} tokens) and its output was cut off.",
        "the model had more to emit than one call allows - the payload is truncated.",
        "rerun the same command - the checkpoint resumes at this stage. If it repeats, raise "
        f"{_MAX_TOKENS_ENV} only deliberately, or route the stage to another provider with "
        "vmodel-spec-app/scripts/switch-model.sh.",
    )


def _guard_limits() -> tuple[int, int, int]:
    def _num(name: str, default: int) -> int:
        raw = _env(name)
        try:
            v = int(raw) if raw else default
        except ValueError:
            v = default
        return v if v > 0 else default

    return (_num("VMODEL_LLM_CALL_TIMEOUT_S", _CALL_TIMEOUT_S),
            _num("VMODEL_LLM_STALL_TIMEOUT_S", _STALL_TIMEOUT_S),
            _num("VMODEL_LLM_MAX_CALLS", _MAX_CALLS))


def _llm_log_sink() -> str | None:
    """VMODEL_LLM_LOG: unset = off (default, zero cost); 'stderr' = console;
    any other value = a file path, one JSON object per line."""
    return _env("VMODEL_LLM_LOG")


def _mask_secrets(value: str) -> str:
    return _re.sub(r"(sk-[A-Za-z0-9_-]{4})[A-Za-z0-9_-]+", r"\1...", str(value))


def _log_llm(event: dict) -> None:
    """One JSON line per LLM call event. Never logs message content or keys -
    metadata, token usage, and provider error payloads only. Logging must
    never break a run."""
    sink = _llm_log_sink()
    if not sink:
        return
    line = json.dumps({**event, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=str)
    try:
        if sink == "stderr":
            print(line, file=sys.stderr)
        else:
            with open(sink, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError:
        pass


def _provider_error_detail(exc) -> dict:
    """Best-effort raw provider error for logs and the error taxonomy: HTTP
    status, error type/code/message, request id, retry-after. The openai SDK
    shape (openai, deepseek) carries .status_code/.body/.response; local
    ollama errors degrade to class + message. Secret-shaped strings masked."""
    out = {"class": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if status is not None:
        out["http_status"] = status
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for h, key in (("x-request-id", "request_id"), ("retry-after", "retry_after_s")):
        try:
            v = headers.get(h)
        except Exception:
            v = None
        if v:
            out[key] = v
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        for k in ("type", "code", "param", "message"):
            if err.get(k) is not None:
                out[k] = _mask_secrets(str(err[k]))[:600]
    if "message" not in out:
        out["message"] = _mask_secrets(str(exc))[:600]
    return out


def _usage_of(resp) -> dict:
    usage = getattr(resp, "usage_metadata", None) or {}
    return {k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens") if k in usage}


def _rerun_guidance(stage: str) -> str:
    """The rerun instruction that is TRUE for this invocation: an explicit
    --from <stage> forces a re-run and discards preserved per-item work, so
    after a failure the right command drops --from."""
    frm = os.environ.get("SPEC_GENERATOR_FROM")
    if frm:
        return (f"rerun the command WITHOUT '--from {frm}' ('--from' forces a re-run from '{frm}' and "
                "discards finished per-item work) - the checkpoint resumes at this stage and finished "
                "items are skipped")
    return "rerun the same command - the checkpoint resumes at this stage"

def _guard_enter(stage: str) -> None:
    """One LLM call is starting: charge it against the run budget."""
    global _CALLS_USED
    _CALLS_USED += 1
    if _CALLS_USED > _guard_limits()[2]:
        cfg = for_stage(stage)
        from .errors import UserFacingError

        raise UserFacingError(
            f"stage '{stage}': this run hit its LLM call budget ({_guard_limits()[2]} calls) and was stopped.",
            "the budget bounds what one run can cost and catches retry loops; the "
            f"{cfg.provider}/{cfg.model} calls before it all completed.",
            "rerun the same command - the checkpoint resumes at this stage. If the run keeps "
            "hitting the budget a stage is looping: report it. Raise the ceiling with "
            "VMODEL_LLM_MAX_CALLS only deliberately.",
        )


def _guard_killed(stage: str, kind: str, waited: float, limit: int):
    cfg = for_stage(stage)
    from .errors import UserFacingError

    return UserFacingError(
        f"stage '{stage}': the {cfg.provider}/{cfg.model} call was killed - {kind} "
        f"(limit {limit}s, waited {waited:.0f}s).",
        "a runaway generation or a hung connection; the call was making no useful progress.",
        "rerun the same command - the checkpoint resumes at this stage. If it repeats, check "
        "the provider is healthy (for ollama: ollama ps + server logs), raise "
        "VMODEL_LLM_CALL_TIMEOUT_S / VMODEL_LLM_STALL_TIMEOUT_S, or route the stage to another "
        "provider with vmodel-spec-app/scripts/switch-model.sh.",
    )


def _run_with_deadline(fn, timeout_s: int, stage: str):
    """Wall-clock deadline around one blocking call. The worker thread cannot
    be force-killed in Python, so the client ALSO gets the same timeout (see
    build_chat_model) - the socket dies, the thread unwinds, and the caller
    gets an informative error instead of a hang."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError from None


@dataclass(frozen=True)
class StageModel:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096


_STAGE_DEFAULTS: dict[str, StageModel] = {
    "residue_judge": StageModel("ollama", "qwen2.5-coder:14b"),
}

_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": None,
    "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}
_API_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "ollama": None}


def for_stage(stage: str) -> StageModel:
    if stage not in _STAGE_DEFAULTS:
        raise KeyError(f"unknown stage {stage!r}")
    cfg = _STAGE_DEFAULTS[stage]
    override = _env(f"VERIFICATION_LOOP_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def _preflight(cfg: "StageModel", stage: str) -> None:
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not _env(env):
        raise UserFacingError(
            f"stage '{stage}' is routed to provider '{cfg.provider}' (model {cfg.model}), but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... or run this stage offline via Ollama: "
            f"VERIFICATION_LOOP_MODEL_{stage.upper()}=ollama:qwen2.5-coder:14b (note: a local judge is fine "
            "for development; founder-facing verdicts should come from the strong model).",
        )


def invoke_checked(chat_model, messages, stage: str = "residue_judge", item: str | None = None):
    """chat_model.invoke under the guardrails (budget + deadline), with the
    known pains translated to human language. Every call is logged when
    VMODEL_LLM_LOG is set (start/end/error, latency, token usage, raw provider
    error payload - never message content, never keys)."""
    from .errors import UserFacingError, translate_llm_error

    cfg = for_stage(stage)
    _guard_enter(stage)
    call_timeout = _guard_limits()[0]
    base = {"provider": cfg.provider, "model": cfg.model, "stage": stage,
            **({"item": item} if item else {})}
    _log_llm({"event": "llm_call_start", **base})
    t0 = time.monotonic()
    try:
        resp = _run_with_deadline(lambda: chat_model.invoke(messages), call_timeout, stage)
        _check_truncated(resp, stage, cfg)
        _log_llm({"event": "llm_call_end", **base, "latency_s": round(time.monotonic() - t0, 2),
                  "usage": _usage_of(resp)})
        return resp
    except TimeoutError:
        _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                  "error": {"kind": "timeout", "limit_s": call_timeout}})
        raise _guard_killed(stage, "no completion within the call budget",
                            call_timeout, call_timeout) from None
    except UserFacingError:
        raise
    except Exception as e:  # noqa: BLE001
        _log_llm({"event": "llm_call_error", **base, "latency_s": round(time.monotonic() - t0, 2),
                  "error": _provider_error_detail(e)})
        friendly = translate_llm_error(e, stage, cfg.provider, cfg.model, _BASE_URLS.get(cfg.provider))
        if friendly is not None:
            raise friendly from e
        raise


def build_chat_model(stage: str = "residue_judge"):
    cfg = for_stage(stage)
    _preflight(cfg, stage)
    if cfg.provider in ("deepseek", "openai", "ollama"):
        from langchain_openai import ChatOpenAI

        env = _API_KEY_ENV[cfg.provider]
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature,
            timeout=_guard_limits()[0], max_tokens=_max_tokens_limit(cfg.max_tokens),
                          base_url=_BASE_URLS[cfg.provider],
                          api_key=(_env(env) or "" if env else "") or "ollama")
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature,
            timeout=_guard_limits()[0], max_tokens=_max_tokens_limit(cfg.max_tokens),
                             api_key=_env(_API_KEY_ENV["anthropic"]) or "")
    raise ValueError(cfg.provider)

