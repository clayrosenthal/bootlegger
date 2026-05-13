import io
import json
import threading

from fastapi import UploadFile
from starlette.datastructures import Headers

from bootlegger.transcribe import _sse, handle_transcription
from tests.conftest import make_wav_bytes


def _upload(data: bytes, filename: str = "audio.wav") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers=Headers({"content-type": "audio/wav"}),
    )


def test_sse_format_round_trips_through_json_load():
    line = _sse({"type": "transcript.text.delta", "delta": "hi"})
    assert line.startswith("data: ")
    assert line.endswith("\n\n")
    payload = json.loads(line[len("data: ") : -2])
    assert payload == {"type": "transcript.text.delta", "delta": "hi"}


def test_handle_transcription_json_default(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = handle_transcription(fake_transcriber, threading.Lock(), _upload(wav), "en", "json")
    assert response.status_code == 200
    assert json.loads(response.body) == {"text": "hello world goodbye"}


def test_handle_transcription_text(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = handle_transcription(fake_transcriber, threading.Lock(), _upload(wav), "en", "text")
    assert response.body.decode() == "hello world goodbye"


def test_handle_transcription_verbose_json(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = handle_transcription(
        fake_transcriber, threading.Lock(), _upload(wav), "fr", "verbose_json"
    )
    body = json.loads(response.body)
    assert body["language"] == "fr"
    assert body["text"] == "hello world goodbye"
    assert len(body["segments"]) == 2


def test_handle_transcription_srt_content_type(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = handle_transcription(fake_transcriber, threading.Lock(), _upload(wav), "en", "srt")
    assert "text/plain" in response.headers["content-type"]
    assert b"hello world" in response.body


def test_handle_transcription_vtt_content_type(fake_transcriber):
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    response = handle_transcription(fake_transcriber, threading.Lock(), _upload(wav), "en", "vtt")
    assert "text/vtt" in response.headers["content-type"]
    assert response.body.startswith(b"WEBVTT")
