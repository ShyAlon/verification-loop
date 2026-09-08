"""Model routing for the ONE LLM call in verification: the semantic-residue
judge. Providers: deepseek (default), openai, anthropic, ollama (local, $0 -
a local judge is fine for development; verdicts shown to founders should come
from the strong model, the verdicts ARE the product).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StageModel:
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096


_STAGE_DEFAULTS: dict[str, StageModel] = {
    "residue_judge": StageModel("deepseek", "deepseek-chat"),
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
    override = os.environ.get(f"VERIFICATION_LOOP_MODEL_{stage.upper()}")
    if override:
        provider, _, model = override.partition(":")
        return StageModel(provider, model, cfg.temperature, cfg.max_tokens)
    return cfg


def build_chat_model(stage: str = "residue_judge"):
    cfg = for_stage(stage)
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
