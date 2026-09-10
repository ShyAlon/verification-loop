"""Guardrails around every LLM call: run-wide call budget + per-call deadline,
informative kills, never silent hangs."""
import time

import pytest

import verification_loop.model_config as mc
from verification_loop.errors import UserFacingError


class FakeModel:
    def __init__(self, text="ok", invoke_sleep_s=0.0):
        self._text, self._invoke_sleep = text, invoke_sleep_s

    def invoke(self, messages):
        time.sleep(self._invoke_sleep)
        return self._text


@pytest.fixture(autouse=True)
def _clean_guard_env(monkeypatch):
    monkeypatch.setattr(mc, "_CALLS_USED", 0)
    for v in ("VMODEL_LLM_CALL_TIMEOUT_S", "VMODEL_LLM_MAX_CALLS", "VMODEL_LLM_MAX_TOKENS"):
        monkeypatch.delenv(v, raising=False)


def test_invoke_under_deadline_returns_normally():
    assert mc.invoke_checked(FakeModel("done"), []) == "done"


def test_slow_invoke_is_killed_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "1")
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(invoke_sleep_s=5), [])
    err = ei.value.render()
    assert "killed" in err and "'residue_judge'" in err and "qwen2.5-coder:14b" in err


def test_run_call_budget_trips_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_MAX_CALLS", "1")
    mc.invoke_checked(FakeModel(), [])
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(), [])
    assert "budget" in ei.value.render()


class _Resp:
    def __init__(self, content, reason=None):
        self.content = content
        self.response_metadata = {} if reason is None else {"done_reason": reason}


def test_generation_length_ceiling_defaults_to_the_stage_ceiling(monkeypatch):
    assert mc._max_tokens_limit(4096) == 4096
    monkeypatch.setenv("VMODEL_LLM_MAX_TOKENS", "1024")
    assert mc._max_tokens_limit(4096) == 1024
    monkeypatch.setenv("VMODEL_LLM_MAX_TOKENS", "junk")
    assert mc._max_tokens_limit(4096) == 4096  # garbage falls back


def test_truncated_generation_is_named_not_silently_passed():
    class M:
        def invoke(self, messages):
            return _Resp('{"partial":', "length")  # ollama done_reason / openai finish_reason

    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(M(), [], "residue_judge")
    err = ei.value.render()
    assert "generation-length ceiling" in err and "4096 tokens" in err
    assert "VMODEL_LLM_MAX_TOKENS" in err and "'residue_judge'" in err


def test_clean_finish_reason_passes():
    class M:
        def invoke(self, messages):
            return _Resp("{}", "stop")

    assert mc.invoke_checked(M(), [], "residue_judge").content == "{}"
