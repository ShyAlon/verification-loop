"""Repo-local .env loading (written by vmodel-spec-app/scripts/switch-model.sh):
file fills unset vars, real environment always wins, loaded once."""
import importlib

import verification_loop.model_config as mc


def _fresh(monkeypatch, tmp_path, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(content)
    for v in ("VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE", "VERIFICATION_LOOP_MODEL_NONE"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(mc, "_DOTENV", None)


def test_dotenv_applies_stage_overrides(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE=deepseek:deepseek-v4-flash\n")
    assert mc.for_stage("residue_judge").model == "deepseek-v4-flash"
    import pytest
    with pytest.raises(KeyError):
        mc.for_stage("no_such_stage")  # residue_judge is the only LLM stage


def test_real_environment_wins_over_dotenv(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE=deepseek:deepseek-v4-flash\n")
    monkeypatch.setenv("VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE", "ollama:qwen2.5-coder:7b")
    assert mc.for_stage("residue_judge").model == "qwen2.5-coder:7b"


def test_missing_or_malformed_dotenv_is_inert(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "# comment\nNO_EQUALS_SIGN\n\n")
    assert mc.for_stage("residue_judge").provider == "ollama"
    (tmp_path / ".env").unlink()
    monkeypatch.setattr(mc, "_DOTENV", None)
    assert mc.for_stage("residue_judge").provider == "ollama"

def test_env_helper_reads_api_key_from_dotenv(monkeypatch, tmp_path):
    """The .env must carry API keys too (switch-model.sh writes them there),
    not just model routing - real environment still wins."""
    _fresh(monkeypatch, tmp_path, "OPENAI_API_KEY=sk-fixture-from-dotenv\n")
    assert mc._env("OPENAI_API_KEY") == "sk-fixture-from-dotenv"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-env-wins")
    assert mc._env("OPENAI_API_KEY") == "sk-real-env-wins"
    assert mc._env("DEEPSEEK_API_KEY") is None
