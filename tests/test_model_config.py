import pytest

from verification_loop import model_config


def test_ollama_override_routes(monkeypatch):
    monkeypatch.setenv("VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE", "ollama:qwen2.5-coder:14b")
    cfg = model_config.for_stage("residue_judge")
    assert (cfg.provider, cfg.model) == ("ollama", "qwen2.5-coder:14b")


def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        model_config.for_stage("vibes")
