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
    for v in ("VMODEL_LLM_CALL_TIMEOUT_S", "VMODEL_LLM_MAX_CALLS"):
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
