import pytest

from verification_loop import model_config


def test_ollama_override_routes(monkeypatch):
    monkeypatch.setenv("VERIFICATION_LOOP_MODEL_RESIDUE_JUDGE", "ollama:qwen2.5-coder:14b")
    cfg = model_config.for_stage("residue_judge")
    assert (cfg.provider, cfg.model) == ("ollama", "qwen2.5-coder:14b")


def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        model_config.for_stage("vibes")


# ---- error handling ----

def test_preflight_names_missing_remote_key(monkeypatch):
    import pytest

    from verification_loop.errors import UserFacingError
    from verification_loop.model_config import build_chat_model

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(UserFacingError) as ei:
        build_chat_model("residue_judge")
    assert "DEEPSEEK_API_KEY" in ei.value.render()


def test_unknown_stage_still_loud():
    import pytest

    from verification_loop.model_config import for_stage

    with pytest.raises(KeyError):
        for_stage("no_such_stage")


def test_ollama_embed_down_is_friendly(monkeypatch):
    import httpx
    import pytest

    from verification_loop.embeddings_trace import ollama_embed
    from verification_loop.errors import UserFacingError

    real_post = httpx.post
    def boom(*a, **k):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "post", boom)
    with pytest.raises(UserFacingError) as ei:
        ollama_embed()(["x"])
    assert "ollama serve" in ei.value.render()
    monkeypatch.setattr(httpx, "post", real_post)
