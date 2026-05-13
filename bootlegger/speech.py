import io
import struct
import threading
import wave
from typing import Optional, Union

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    input: str = Field(..., max_length=4096)
    model: str
    voice: Union[str, dict] = "alloy"
    instructions: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = None
    stream_format: Optional[str] = None
    language: Optional[str] = None


_FORMAT_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

_PYDUB_FORMATS = {
    "mp3": ("mp3", None),
    "opus": ("ogg", "libopus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", None),
}

_OPENAI_VOICE_TO_KOKORO = {
    "alloy": "kokoro_af_alloy",
    "echo": "kokoro_am_echo",
    "nova": "kokoro_af_nova",
    "onyx": "kokoro_am_onyx",
}


def _samples_to_int16(samples) -> bytes:
    out = bytearray(len(samples) * 2)
    for i, s in enumerate(samples):
        if s > 1.0:
            s = 1.0
        elif s < -1.0:
            s = -1.0
        struct.pack_into("<h", out, i * 2, int(s * 32767.0))
    return bytes(out)


def _encode_wav(samples, sample_rate: int) -> bytes:
    pcm = _samples_to_int16(samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _encode_with_pydub(samples, sample_rate: int, fmt: str) -> bytes:
    from pydub import AudioSegment

    pydub_fmt, codec = _PYDUB_FORMATS[fmt]
    pcm = _samples_to_int16(samples)
    seg = AudioSegment(
        data=pcm, sample_width=2, frame_rate=sample_rate, channels=1
    )
    buf = io.BytesIO()
    if codec is not None:
        seg.export(buf, format=pydub_fmt, codec=codec)
    else:
        seg.export(buf, format=pydub_fmt)
    return buf.getvalue()


def _encode(samples, sample_rate: int, fmt: str) -> bytes:
    if fmt == "wav":
        return _encode_wav(samples, sample_rate)
    if fmt == "pcm":
        return _samples_to_int16(samples)
    if fmt in _PYDUB_FORMATS:
        return _encode_with_pydub(samples, sample_rate, fmt)
    raise HTTPException(
        status_code=400, detail=f"Unsupported response_format: {fmt}"
    )


def _resolve_voice(voice: Union[str, dict, None]) -> Optional[str]:
    if voice is None:
        return None
    if isinstance(voice, dict):
        return voice.get("id")
    if voice.startswith(("kokoro_", "piper_")):
        return voice
    return _OPENAI_VOICE_TO_KOKORO.get(voice, voice)


_TTS_CACHE: dict = {}
_TTS_CACHE_LOCK = threading.Lock()


def _get_tts(language: str, voice: Optional[str]):
    from moonshine_voice import TextToSpeech, list_tts_languages

    key = (language, voice)
    with _TTS_CACHE_LOCK:
        tts = _TTS_CACHE.get(key)
        if tts is None:
            if language not in list_tts_languages():
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported language: {language}",
                )
            tts = TextToSpeech(language=language, voice=voice)
            _TTS_CACHE[key] = tts
    return tts


def handle_speech(req: SpeechRequest, default_language: str) -> Response:
    fmt = (req.response_format or "mp3").lower()
    if fmt not in _FORMAT_MEDIA_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported response_format: {fmt}"
        )

    language = (req.language or default_language).replace("_", "-")
    voice = _resolve_voice(req.voice)

    try:
        tts = _get_tts(language, voice)
        kwargs = {}
        if req.speed is not None:
            kwargs["speed"] = req.speed
        samples, sample_rate = tts.synthesize(req.input, **kwargs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    audio_bytes = _encode(samples, sample_rate, fmt)
    return Response(content=audio_bytes, media_type=_FORMAT_MEDIA_TYPES[fmt])
