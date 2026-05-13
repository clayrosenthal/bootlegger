from bootlegger.config import Settings


def test_defaults():
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.language == "en"
    assert s.model_arch is None
    assert s.api_prefix == "/v1"
    assert s.tts_language == "en-us"
    assert s.tts_voice is None


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("BOOTLEGGER_PORT", "9000")
    monkeypatch.setenv("BOOTLEGGER_LANGUAGE", "es")
    monkeypatch.setenv("BOOTLEGGER_TTS_VOICE", "kokoro_af_nova")
    s = Settings()
    assert s.port == 9000
    assert s.language == "es"
    assert s.tts_voice == "kokoro_af_nova"


def test_model_copy_update_round_trips():
    s = Settings()
    s2 = s.model_copy(update={"port": 1234, "host": "127.0.0.1"})
    assert s2.port == 1234
    assert s2.host == "127.0.0.1"
    assert s.port == 8000  # unchanged
