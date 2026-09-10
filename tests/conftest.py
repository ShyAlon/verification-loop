import pytest

import verification_loop.model_config as mc
@pytest.fixture(autouse=True)
def _reset_llm_call_budget(monkeypatch):
    """Each test is one run: the run-wide LLM call budget starts clean."""
    monkeypatch.setattr(mc, "_CALLS_USED", 0)
