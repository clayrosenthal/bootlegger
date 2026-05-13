import io
import struct
import wave

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from bootlegger.speech import (
    SpeechRequest,
    _encode,
    _encode_wav,
    _resolve_voice,
    _samples_to_int16,
    _split_sentences,
    _streaming_wav_header,
)


def test_resolve_voice_none():
    assert _resolve_voice(None) is None


def test_resolve_voice_dict_id():
    assert _resolve_voice({"id": "voice_xyz"}) == "voice_xyz"


def test_resolve_voice_dict_missing_id():
    assert _resolve_voice({}) is None


def test_resolve_voice_openai_alias_maps_to_kokoro():
    assert _resolve_voice("alloy") == "kokoro_af_alloy"
    assert _resolve_voice("echo") == "kokoro_am_echo"
    assert _resolve_voice("nova") == "kokoro_af_nova"
    assert _resolve_voice("onyx") == "kokoro_am_onyx"


def test_resolve_voice_pass_through_kokoro_prefix():
    assert _resolve_voice("kokoro_custom") == "kokoro_custom"


def test_resolve_voice_pass_through_piper_prefix():
    assert _resolve_voice("piper_custom") == "piper_custom"


def test_resolve_voice_unknown_alias_returned_unchanged():
    assert _resolve_voice("custom-voice") == "custom-voice"


def test_split_sentences_basic():
    assert _split_sentences("Hi there. How are you? I'm fine!") == [
        "Hi there.",
        "How are you?",
        "I'm fine!",
    ]


def test_split_sentences_no_terminators_returns_whole_input():
    assert _split_sentences("just one chunk") == ["just one chunk"]


def test_split_sentences_strips_whitespace_and_empty():
    assert _split_sentences("  one.   two.  ") == ["one.", "two."]


def test_split_sentences_only_whitespace_returns_empty_string_in_list():
    # Implementation falls back to [text.strip()] when split produced nothing.
    assert _split_sentences("   ") == [""]


def test_samples_to_int16_clamps_and_quantizes():
    pcm = _samples_to_int16([0.0, 1.0, -1.0, 2.0, -2.0])
    values = [struct.unpack_from("<h", pcm, i * 2)[0] for i in range(5)]
    assert values[0] == 0
    assert values[1] == 32767
    assert values[2] == -32767
    assert values[3] == 32767  # clamp >1
    assert values[4] == -32767  # clamp <-1


def test_encode_wav_is_valid_riff(wav_bytes):
    out = _encode_wav([0.0] * 8000, 16000)
    with wave.open(io.BytesIO(out), "rb") as r:
        assert r.getnchannels() == 1
        assert r.getsampwidth() == 2
        assert r.getframerate() == 16000
        assert r.getnframes() == 8000


def test_encode_pcm_returns_raw_bytes():
    pcm = _encode([0.0, 0.5], 16000, "pcm")
    assert isinstance(pcm, bytes)
    assert len(pcm) == 4  # 2 samples * 2 bytes


def test_encode_unknown_format_raises_http_400():
    with pytest.raises(HTTPException) as exc:
        _encode([0.0], 16000, "bogus")
    assert exc.value.status_code == 400


def test_streaming_wav_header_layout():
    header = _streaming_wav_header(24000)
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
    assert header[12:16] == b"fmt "
    # Sample rate field at offset 24.
    assert struct.unpack_from("<I", header, 24)[0] == 24000
    # Byte rate = sample_rate * 2.
    assert struct.unpack_from("<I", header, 28)[0] == 48000
    # Streaming-WAV size placeholders.
    assert struct.unpack_from("<I", header, 4)[0] == 0xFFFFFFFF
    assert struct.unpack_from("<I", header, 40)[0] == 0xFFFFFFFF


def test_speech_request_defaults():
    req = SpeechRequest(input="hello", model="tts-1")
    assert req.voice == "alloy"
    assert req.response_format == "mp3"
    assert req.speed is None
    assert req.stream_format is None
    assert req.language is None


def test_speech_request_accepts_dict_voice():
    req = SpeechRequest(input="hello", model="tts-1", voice={"id": "voice_x"})
    assert req.voice == {"id": "voice_x"}


def test_speech_request_input_max_length():
    SpeechRequest(input="x" * 4096, model="tts-1")
    with pytest.raises(ValidationError):
        SpeechRequest(input="x" * 4097, model="tts-1")
