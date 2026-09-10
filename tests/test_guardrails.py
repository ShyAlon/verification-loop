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
    for v in ("VMODEL_LLM_CALL_TIMEOUT_S", "VMODEL_LLM_MAX_CALLS", "VMODEL_LLM_MAX_TOKENS", "VMODEL_LLM_LOG", "VMODEL_LLM_HEARTBEAT_S", "VMODEL_OLLAMA_THINK", "VMODEL_OLLAMA_NUM_CTX"):
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


class RateLimitError(Exception):
    """Mimics the openai SDK exception shape (status/body/response.headers)."""

    def __init__(self, message, err_type, code=None, headers=None):
        super().__init__(message)
        self.status_code = 429
        self.body = {"error": {"message": message, "type": err_type, "code": code}}
        self.response = type("R", (), {"headers": headers or {}})()


def _read_log(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_llm_logging_off_by_default(tmp_path):
    log = tmp_path / "llm.jsonl"
    mc.invoke_checked(type("M", (), {"invoke": lambda self, m: type("R", (), {})()})(), [], "residue_judge")
    assert not log.exists()


def test_llm_logging_file_sink_start_end_usage(tmp_path, monkeypatch):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))

    class M:
        def invoke(self, messages):
            return type("R", (), {"usage_metadata": {"input_tokens": 5, "output_tokens": 3,
                                                     "total_tokens": 8}})()

    mc.invoke_checked(M(), [], "residue_judge", item="IT-1")
    recs = _read_log(log)
    assert [r["event"] for r in recs] == ["llm_call_start", "llm_call_end"]
    assert recs[0]["provider"] == "ollama" and recs[0]["stage"] == "residue_judge" and recs[0]["item"] == "IT-1"
    assert recs[1]["usage"]["total_tokens"] == 8 and recs[1]["latency_s"] >= 0
    assert "messages" not in recs[0] and "prompt" not in recs[0]  # no prompt content logged


def test_llm_logging_stderr_sink(monkeypatch, capsys):
    monkeypatch.setenv("VMODEL_LLM_LOG", "stderr")
    mc.invoke_checked(type("M", (), {"invoke": lambda self, m: type("R", (), {})()})(), [], "residue_judge")
    assert '"llm_call_start"' in capsys.readouterr().err


def test_llm_logging_error_record_masks_keys(tmp_path, monkeypatch):
    log = tmp_path / "llm.jsonl"
    monkeypatch.setenv("VMODEL_LLM_LOG", str(log))
    secret = "sk-abcdVERYSECRETKEY12345"
    exc = RateLimitError(f"quota exceeded for key {secret}", "insufficient_quota",
                              code="insufficient_quota",
                              headers={"x-request-id": "req_test", "retry-after": "20"})

    class M:
        def invoke(self, messages):
            raise exc

    with pytest.raises(UserFacingError):
        mc.invoke_checked(M(), [], "residue_judge")
    rec = [r for r in _read_log(log) if r["event"] == "llm_call_error"][0]
    assert rec["error"]["http_status"] == 429
    assert rec["error"]["code"] == "insufficient_quota"
    assert rec["error"]["request_id"] == "req_test"
    assert rec["error"]["retry_after_s"] == "20"
    assert secret not in log.read_text()  # masked


def test_insufficient_quota_is_billing_not_throughput():
    from verification_loop.errors import translate_llm_error

    exc = RateLimitError("You exceeded your current quota, please check your plan and billing details",
                              "insufficient_quota", code="insufficient_quota")
    err = translate_llm_error(exc, "residue_judge", "openai", "gpt-x", None)
    r = err.render()
    assert "no usable quota" in r and "billing" in r and "waiting changes nothing" in r
    assert "switch-model.sh ollama" in r


def test_true_rate_limit_names_throughput_and_retry_after():
    from verification_loop.errors import translate_llm_error

    exc = RateLimitError("Rate limit reached for gpt-x on requests per min (RPM): max 50",
                              "rate_limit_exceeded", headers={"retry-after": "20"})
    err = translate_llm_error(exc, "residue_judge", "openai", "gpt-x", None)
    r = err.render()
    assert "throughput" in r and "20s" in r


def test_generic_429_points_at_raw_log():
    from verification_loop.errors import translate_llm_error

    exc = RateLimitError("some other 429 situation", "some_type")
    err = translate_llm_error(exc, "residue_judge", "openai", "gpt-x", None)
    assert "VMODEL_LLM_LOG" in err.render()


def test_403_speaks_of_tier_access():
    from verification_loop.errors import translate_llm_error

    exc = type("PermissionDeniedError", (Exception,), {"status_code": 403})("error code: 403 - denied")
    err = translate_llm_error(exc, "residue_judge", "openai", "gpt-x", None)
    r = err.render()
    assert "denied access (403)" in r and "tier" in r


def test_heartbeat_prints_during_silent_call_then_stops(monkeypatch, capsys):
    monkeypatch.setenv("VMODEL_LLM_HEARTBEAT_S", "0.05")

    class M:
        def invoke(self, messages):
            time.sleep(0.2)
            return type("R", (), {})()

    mc.invoke_checked(M(), [], "residue_judge")
    err = capsys.readouterr().err
    assert "still working - stage 'residue_judge' call" in err
    time.sleep(0.15)  # ~3 more intervals: no further beats once the call returned
    assert "still working" not in capsys.readouterr().err


def test_heartbeat_disabled_with_zero(monkeypatch, capsys):
    monkeypatch.setenv("VMODEL_LLM_HEARTBEAT_S", "0")

    class M:
        def invoke(self, messages):
            time.sleep(0.1)
            return type("R", (), {})()

    mc.invoke_checked(M(), [], "residue_judge")
    assert "still working" not in capsys.readouterr().err


def test_describe_models_names_resolved_provider_and_model():
    s = mc.describe_models()
    assert "ollama/qwen2.5-coder:14b" in s and "residue_judge=" in s


def test_ollama_thinking_off_by_default(monkeypatch):
    monkeypatch.delenv("VMODEL_OLLAMA_THINK", raising=False)
    m = mc.build_chat_model("residue_judge")
    assert m.extra_body["think"] is False


def test_ollama_thinking_knob_reenables(monkeypatch):
    monkeypatch.setenv("VMODEL_OLLAMA_THINK", "1")
    m = mc.build_chat_model("residue_judge")
    assert not m.extra_body or "think" not in m.extra_body


def test_ollama_num_ctx_default_and_knob(monkeypatch):
    monkeypatch.delenv("VMODEL_OLLAMA_NUM_CTX", raising=False)
    m = mc.build_chat_model("residue_judge")
    assert m.extra_body["num_ctx"] == 32768
    monkeypatch.setenv("VMODEL_OLLAMA_NUM_CTX", "16384")
    m = mc.build_chat_model("residue_judge")
    assert m.extra_body["num_ctx"] == 16384
