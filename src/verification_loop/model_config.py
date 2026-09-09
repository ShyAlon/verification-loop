"""Model routing for the ONE LLM call in verification: the semantic-residue
judge. Providers: ollama (DEFAULT - local-first, ADR-0018), deepseek/openai/anthropic (opt-in),
a local judge is fine for development; verdicts shown to founders should come
from the strong model, the verdicts ARE the product).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_DOTENV: dict[str, str] | None = None


def _dotenv() -> dict[str, str]:
    """Repo-local .env (written by vmodel-spec-app/scripts/switch-model.sh),
    parsed once per process. Real environment variables always win over the
    file, and os.environ is never mutated - no dependency, no side effects."""
    global _DOTENV
    if _DOTENV is not None:
        return _DOTENV
    _DOTENV = {}
    from pathlib import Path

    p = Path(".env")
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            _DOTENV[key.strip()] = value.strip().strip('"').strip("'")
    return _DOTENV


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
    override = os.environ.get(f"VERIFICATION_LOOP_MODEL_{stage.upper()}") or _dotenv().get(f"VERIFICATION_LOOP_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def _preflight(cfg: "StageModel", stage: str) -> None:
    from .errors import UserFacingError

    env = _API_KEY_ENV.get(cfg.provider)
    if env and not os.environ.get(env):
        raise UserFacingError(
            f"stage '{stage}' is routed to provider '{cfg.provider}' (model {cfg.model}), but {env} is not set.",
            "no API key in the environment.",
            f"export {env}=... or run this stage offline via Ollama: "
            f"VERIFICATION_LOOP_MODEL_{stage.upper()}=ollama:qwen2.5-coder:14b (note: a local judge is fine "
            "for development; founder-facing verdicts should come from the strong model).",
        )


def invoke_checked(chat_model, messages, stage: str = "residue_judge"):
    from .errors import translate_llm_error

    cfg = for_stage(stage)
    try:
        return chat_model.invoke(messages)
    except Exception as e:  # noqa: BLE001
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
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                          base_url=_BASE_URLS[cfg.provider],
                          api_key=(os.environ.get(env, "") if env else "") or "ollama")
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                             api_key=os.environ.get(_API_KEY_ENV["anthropic"], ""))
    raise ValueError(cfg.provider)

