"""Human-readable failure handling.

Philosophy: a failure the USER can fix (no API key, Ollama down, model not
pulled, unparseable model output, bad input file) gets a three-part message -
what happened, the likely cause, the exact next action - and exits 1. A
genuine bug (anything unrecognized) still raises: we translate known pains,
we do not wallpaper over real bugs. Unpriced REMOTE LLM calls still crash
loudly on purpose (telemetry guard) - but even that message names the fix.
"""
from __future__ import annotations

import json
import sys


class UserFacingError(Exception):
    def __init__(self, what: str, cause: str | None = None, next_action: str | None = None):
        self.what, self.cause, self.next_action = what, cause, next_action
        super().__init__(what)

    def render(self) -> str:
        parts = [f"Error: {self.what}"]
        if self.cause:
            parts.append(f"Likely cause: {self.cause}")
        if self.next_action:
            parts.append(f"Next: {self.next_action}")
        return "\n".join(parts)


def _is_ollama_ctx(context: str) -> bool:
    return "ollama" in context.lower()


def translate_llm_error(exc: Exception, stage: str, provider: str, model: str, base_url: str | None) -> UserFacingError | None:
    """Map the common LLM-call failures to UserFacingError. None = not ours."""
    name = type(exc).__name__
    text = str(exc).lower()
    ctx = f"{provider} {base_url or ''}"

    if name in ("APIConnectionError", "ConnectError", "ConnectTimeout") or "connection" in text:
        if _is_ollama_ctx(ctx):
            return UserFacingError(
                f"stage '{stage}' could not reach the local Ollama server at {base_url}.",
                "Ollama is not running (or OLLAMA_BASE_URL points elsewhere).",
                "run `ollama serve` (or start the Ollama app) and retry.",
            )
        return UserFacingError(
            f"stage '{stage}' could not reach provider '{provider}'.",
            "network problem or provider outage.",
            "check connectivity / the provider status page, then retry.",
        )
    if name == "AuthenticationError" or "401" in text or "invalid api key" in text:
        env = {"deepseek": "DEEPSEEK_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "API key")
        return UserFacingError(
            f"stage '{stage}' was rejected by '{provider}' (authentication).",
            f"missing or invalid key - the environment variable {env} is empty or wrong.",
            f"export {env}=... (or run locally: {stage.upper()} via ollama - see README 'Local runs with Ollama').",
        )
    if name == "NotFoundError" or ("404" in text and "model" in text) or "not found" in text:
        if _is_ollama_ctx(ctx):
            return UserFacingError(
                f"stage '{stage}': Ollama does not have model '{model}'.",
                "the model was never pulled on this machine.",
                f"run `ollama pull {model}` and retry.",
            )
        return UserFacingError(
            f"stage '{stage}': provider '{provider}' does not know model '{model}'.",
            "typo in the model override, the model was renamed/retired, or the key's tier has "
            "no access to this model.",
            "check the *_MODEL_* env override against the provider's model list, and the key's "
            "tier/model access in the provider console.",
        )
    if name == "PermissionDeniedError" or "error code: 403" in text:
        from .model_config import _provider_error_detail

        detail = _provider_error_detail(exc)
        return UserFacingError(
            f"stage '{stage}': '{provider}' denied access (403).",
            f"the key authenticated, but this model or feature may not be enabled on its tier. "
            f"detail: {str(detail.get('message', ''))[:200]}",
            "check the key's tier/model access in the provider console - or route the stage to "
            "ollama: vmodel-spec-app/scripts/switch-model.sh ollama.",
        )
    if name == "RateLimitError" or "429" in text:
        # 429 is three different failures wearing one status: split them,
        # because the right action for each is different (2026-09-10: a key
        # with no quota was told to "wait a minute").
        from .model_config import _provider_error_detail, _rerun_guidance

        detail = _provider_error_detail(exc)
        blob = " ".join(str(detail.get(k, "")) for k in ("type", "code", "message")).lower()
        short = str(detail.get("message", ""))[:140]
        if "insufficient_quota" in blob or "plan and billing" in blob:
            return UserFacingError(
                f"stage '{stage}': '{provider}' rejected the call - the key has no usable quota.",
                f"insufficient_quota is billing/credits, not throughput - waiting changes nothing. "
                f"provider said: {short}",
                "add a payment method or credits in the provider console - or switch back to local: "
                "vmodel-spec-app/scripts/switch-model.sh ollama.",
            )
        if "rate limit reached" in blob or "requests per" in blob or "tokens per" in blob:
            wait = (f"the provider says retry after {detail['retry_after_s']}s. "
                    if detail.get("retry_after_s") else "")
            return UserFacingError(
                f"stage '{stage}' hit '{provider}' throughput limits.",
                f"true rate limiting - the key is sending faster than its plan allows: {short}",
                f"{wait}{_rerun_guidance(stage)}; persistent? lower parallel load or raise the plan tier.",
            )
        return UserFacingError(
            f"stage '{stage}' hit '{provider}' rate limits.",
            f"provider detail: {short or 'unknown'}.",
            f"{_rerun_guidance(stage)}; persistent? check the key's plan. Set VMODEL_LLM_LOG=<path> "
            "for the full raw provider payload.",
        )
    return None


def translate_json_error(exc: Exception, stage: str) -> UserFacingError:
    return UserFacingError(
        f"stage '{stage}': the model returned output that is not valid JSON.",
        "smaller/local models often wrap JSON in prose or drop structure.",
        "retry once; if it repeats, route this stage to a stronger model "
        "(*_MODEL_* env override) - authoring stages earn the strong model (ADR-0004).",
    )


def run_cli(fn) -> None:
    """CLI wrapper: friendly exit for known pains, loud raise for real bugs."""
    try:
        fn()
    except UserFacingError as e:
        print("\n" + e.render(), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Run state up to here is preserved (checkpointer); rerun to resume.", file=sys.stderr)
        sys.exit(130)
    except json.JSONDecodeError as e:
        print("\n" + translate_json_error(e, "cli").render(), file=sys.stderr)
        sys.exit(1)

