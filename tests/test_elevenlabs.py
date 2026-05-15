import io
import json
import threading
from dataclasses import dataclass

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from bootlegger import elevenlabs as el
from tests.conftest import FakeLine, FakeTranscript, make_wav_bytes

# ---------------------------------------------------------------------------
# _parse_output_format
# ---------------------------------------------------------------------------


def test_parse_output_format_default():
    assert el._parse_output_format(None) == ("mp3", None)


def test_parse_output_format_mp3_with_rate_and_bitrate():
    assert el._parse_output_format("mp3_44100_128") == ("mp3", 44100)


def test_parse_output_format_pcm_rate_only():
    assert el._parse_output_format("pcm_16000") == ("pcm", 16000)


def test_parse_output_format_wav():
    assert el._parse_output_format("wav_48000") == ("wav", 48000)


def test_parse_output_format_unknown_prefix_raises():
    with pytest.raises(HTTPException) as exc:
        el._parse_output_format("bogus_44100")
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Voice catalog filtering & pagination
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_catalog(monkeypatch):
    catalog = [
        {
            "voice_id": "kokoro_af_alloy",
            "name": "kokoro_af_alloy",
            "category": "premade",
            "description": None,
            "preview_url": None,
            "labels": {"languages": ["en-us"]},
            "settings": None,
            "created_at_unix": None,
            "favorited_at_unix": None,
            "is_owner": False,
            "_present": True,
        },
        {
            "voice_id": "kokoro_af_nova",
            "name": "kokoro_af_nova",
            "category": "premade",
            "description": None,
            "preview_url": None,
            "labels": {"languages": ["en-us"]},
            "settings": None,
            "created_at_unix": None,
            "favorited_at_unix": None,
            "is_owner": False,
            "_present": False,
        },
        {
            "voice_id": "piper_es_ES-davefx-medium",
            "name": "piper_es_ES-davefx-medium",
            "category": "premade",
            "description": None,
            "preview_url": None,
            "labels": {"languages": ["es-es"]},
            "settings": None,
            "created_at_unix": None,
            "favorited_at_unix": None,
            "is_owner": False,
            "_present": False,
        },
    ]
    monkeypatch.setattr(el, "_get_voice_catalog", lambda: catalog)
    return catalog


def test_list_voices_returns_full_page_with_total(fake_catalog):
    result = el.list_voices(page_size=10)
    assert result["total_count"] == 3
    assert result["has_more"] is False
    assert result["next_page_token"] is None
    assert {v["voice_id"] for v in result["voices"]} == {
        "kokoro_af_alloy",
        "kokoro_af_nova",
        "piper_es_ES-davefx-medium",
    }
    # Internal _present field is stripped.
    assert all("_present" not in v for v in result["voices"])


def test_list_voices_pagination(fake_catalog):
    page1 = el.list_voices(page_size=2)
    assert len(page1["voices"]) == 2
    assert page1["has_more"] is True
    assert page1["next_page_token"] == "2"

    page2 = el.list_voices(page_size=2, next_page_token="2")
    assert len(page2["voices"]) == 1
    assert page2["has_more"] is False
    assert page2["next_page_token"] is None


def test_list_voices_search_matches_language_label(fake_catalog):
    result = el.list_voices(search="es-es")
    assert [v["voice_id"] for v in result["voices"]] == ["piper_es_ES-davefx-medium"]


def test_list_voices_search_matches_id_substring(fake_catalog):
    result = el.list_voices(search="nova")
    assert [v["voice_id"] for v in result["voices"]] == ["kokoro_af_nova"]


def test_list_voices_voice_ids_filter(fake_catalog):
    result = el.list_voices(voice_ids=["kokoro_af_alloy", "missing"])
    assert [v["voice_id"] for v in result["voices"]] == ["kokoro_af_alloy"]


def test_list_voices_sort_descending(fake_catalog):
    result = el.list_voices(sort="name", sort_direction="desc")
    ids = [v["voice_id"] for v in result["voices"]]
    assert ids == sorted(ids, reverse=True)


def test_list_voices_invalid_page_size():
    with pytest.raises(HTTPException) as exc:
        el.list_voices(page_size=0)
    assert exc.value.status_code == 422


def test_list_voices_invalid_page_token(fake_catalog):
    with pytest.raises(HTTPException) as exc:
        el.list_voices(next_page_token="not-an-int")
    assert exc.value.status_code == 422


def test_list_voices_omits_total_when_disabled(fake_catalog):
    result = el.list_voices(include_total_count=False)
    assert "total_count" not in result


def test_voice_category_classification():
    assert el._voice_category("kokoro_af_alloy") == "premade"
    assert el._voice_category("piper_en_US-amy-low") == "premade"
    assert el._voice_category("custom_voice_xyz") == "generated"


# ---------------------------------------------------------------------------
# _format_words
# ---------------------------------------------------------------------------


@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    confidence: float


def test_format_words_none_granularity_returns_empty():
    transcript = FakeTranscript(lines=[FakeLine("hello", 0.0, 1.0)])
    assert el._format_words(transcript, "none") == []


def test_format_words_word_granularity_emits_word_entries():
    line = FakeLine("hello world", 0.0, 1.0)
    line.words = [
        FakeWord("hello", 0.0, 0.5, -0.1),
        FakeWord("world", 0.5, 1.0, -0.2),
    ]
    transcript = FakeTranscript(lines=[line])
    out = el._format_words(transcript, "word")
    assert out == [
        {"text": "hello", "start": 0.0, "end": 0.5, "type": "word", "logprob": -0.1},
        {"text": "world", "start": 0.5, "end": 1.0, "type": "word", "logprob": -0.2},
    ]


def test_format_words_skips_lines_without_word_data():
    line = FakeLine("hi", 0.0, 0.5)
    line.words = None
    transcript = FakeTranscript(lines=[line])
    assert el._format_words(transcript, "word") == []


# ---------------------------------------------------------------------------
# handle_speech_to_text
# ---------------------------------------------------------------------------


def _upload(data: bytes, filename: str = "audio.wav") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": "audio/wav"}),
    )


def test_handle_speech_to_text_basic_response(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = el.handle_speech_to_text(
        fake_transcriber,
        threading.Lock(),
        _upload(wav),
        model_id="scribe_v1",
        language_code=None,
        timestamps_granularity="none",
        default_language="en",
    )
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["text"] == "hello world goodbye"
    assert body["language_code"] == "en"
    assert body["audio_duration_secs"] == 3.5
    assert body["words"] == []


def test_handle_speech_to_text_strips_region_from_default_language(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = el.handle_speech_to_text(
        fake_transcriber,
        threading.Lock(),
        _upload(wav),
        model_id="scribe_v1",
        language_code=None,
        timestamps_granularity="none",
        default_language="en-us",
    )
    assert json.loads(response.body)["language_code"] == "en"


def test_handle_speech_to_text_uses_supplied_language_code(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = el.handle_speech_to_text(
        fake_transcriber,
        threading.Lock(),
        _upload(wav),
        model_id="scribe_v1",
        language_code="fr",
        timestamps_granularity="none",
        default_language="en",
    )
    assert json.loads(response.body)["language_code"] == "fr"


def test_handle_speech_to_text_missing_file_raises_422(fake_transcriber):
    with pytest.raises(HTTPException) as exc:
        el.handle_speech_to_text(
            fake_transcriber,
            threading.Lock(),
            None,
            model_id="scribe_v1",
            language_code=None,
            timestamps_granularity="none",
            default_language="en",
        )
    assert exc.value.status_code == 422


def test_handle_speech_to_text_rejects_unknown_granularity(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    with pytest.raises(HTTPException) as exc:
        el.handle_speech_to_text(
            fake_transcriber,
            threading.Lock(),
            _upload(wav),
            model_id="scribe_v1",
            language_code=None,
            timestamps_granularity="character",
            default_language="en",
        )
    assert exc.value.status_code == 422


def test_handle_speech_to_text_missing_model_id_raises_422(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    with pytest.raises(HTTPException) as exc:
        el.handle_speech_to_text(
            fake_transcriber,
            threading.Lock(),
            _upload(wav),
            model_id="",
            language_code=None,
            timestamps_granularity="none",
            default_language="en",
        )
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def test_create_app_registers_elevenlabs_routes():
    from bootlegger.config import Settings
    from bootlegger.main import create_app

    app = create_app(Settings())
    paths = {route.path for route in app.routes}
    assert "/v2/voices" in paths
    assert "/v1/text-to-speech/{voice_id}" in paths
    assert "/v1/speech-to-text" in paths


# ---------------------------------------------------------------------------
# TextToSpeechRequest model
# ---------------------------------------------------------------------------


def test_text_to_speech_request_max_length():
    el.TextToSpeechRequest(text="x" * 5000)
    with pytest.raises(Exception):
        el.TextToSpeechRequest(text="x" * 5001)


def test_text_to_speech_request_voice_settings_speed():
    req = el.TextToSpeechRequest(text="hi", voice_settings={"speed": 1.5})
    assert req.voice_settings.speed == 1.5
