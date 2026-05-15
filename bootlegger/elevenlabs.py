"""ElevenLabs-compatible API endpoints backed by Moonshine.

Implements a subset of the public ElevenLabs HTTP API:

- ``GET  /v2/voices``                   list/search voices
- ``POST /v1/text-to-speech/{voice_id}`` synthesize speech
- ``POST /v1/speech-to-text``           transcribe audio
"""

import threading
from typing import Any, Optional

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from bootlegger.audio import decode_audio
from bootlegger.speech import _encode, _resolve_voice

# ---------------------------------------------------------------------------
# Output-format parsing for /v1/text-to-speech/{voice_id}
# ---------------------------------------------------------------------------

# ElevenLabs ``output_format`` strings encode "<container>_<sample_rate>[_<bitrate>]".
# Map the container prefix onto a format known to bootlegger's encoder, plus the
# requested sample rate (when specified) so we can resample on the fly later.
_OUTPUT_FORMAT_PREFIXES = {
    "mp3": "mp3",
    "pcm": "pcm",
    "wav": "wav",
    "opus": "opus",
    "flac": "flac",
    "aac": "aac",
}

_FORMAT_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _parse_output_format(output_format: Optional[str]) -> tuple[str, Optional[int]]:
    """Return ``(internal_fmt, sample_rate_hint)``. Defaults to ``mp3`` with no hint."""
    if not output_format:
        return "mp3", None
    parts = output_format.lower().split("_")
    prefix = parts[0]
    fmt = _OUTPUT_FORMAT_PREFIXES.get(prefix)
    if fmt is None:
        raise HTTPException(status_code=400, detail=f"Unsupported output_format: {output_format}")
    sample_rate: Optional[int] = None
    if len(parts) > 1:
        try:
            sample_rate = int(parts[1])
        except ValueError:
            sample_rate = None
    return fmt, sample_rate


# ---------------------------------------------------------------------------
# Voices catalog (cached)
# ---------------------------------------------------------------------------

_VOICES_CACHE_LOCK = threading.Lock()
_VOICES_CACHE: Optional[list[dict[str, Any]]] = None


def _voice_category(voice_id: str) -> str:
    if voice_id.startswith("kokoro_"):
        return "premade"
    if voice_id.startswith("piper_"):
        return "premade"
    return "generated"


def _build_voice_catalog() -> list[dict[str, Any]]:
    """Aggregate Moonshine voices across all supported languages."""
    from moonshine_voice import list_tts_languages, list_tts_voices

    seen: dict[str, dict[str, Any]] = {}
    for lang in list_tts_languages():
        entry = list_tts_voices(lang)
        for state in ("present", "downloadable"):
            for voice_id in entry.get(state, []):
                rec = seen.setdefault(
                    voice_id,
                    {
                        "voice_id": voice_id,
                        "name": voice_id,
                        "category": _voice_category(voice_id),
                        "description": None,
                        "preview_url": None,
                        "labels": {"languages": []},
                        "settings": None,
                        "created_at_unix": None,
                        "favorited_at_unix": None,
                        "is_owner": False,
                        "_present": state == "present",
                    },
                )
                if state == "present":
                    rec["_present"] = True
                rec["labels"]["languages"].append(lang)
    voices = list(seen.values())
    voices.sort(key=lambda v: v["voice_id"])
    return voices


def _get_voice_catalog() -> list[dict[str, Any]]:
    global _VOICES_CACHE
    with _VOICES_CACHE_LOCK:
        if _VOICES_CACHE is None:
            _VOICES_CACHE = _build_voice_catalog()
        return _VOICES_CACHE


def _public_voice(rec: dict[str, Any]) -> dict[str, Any]:
    """Strip internal fields from a voice record before returning to clients."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def _matches_search(rec: dict[str, Any], term: str) -> bool:
    term = term.lower()
    if term in rec["voice_id"].lower():
        return True
    if term in rec["name"].lower():
        return True
    if term in rec["category"].lower():
        return True
    for lang in rec["labels"].get("languages", []):
        if term in lang.lower():
            return True
    return False


def list_voices(
    next_page_token: Optional[str] = None,
    page_size: int = 10,
    search: Optional[str] = None,
    sort: Optional[str] = None,
    sort_direction: Optional[str] = None,
    voice_type: Optional[str] = None,
    category: Optional[str] = None,
    include_total_count: bool = True,
    voice_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=422, detail="page_size must be between 1 and 100")

    voices = _get_voice_catalog()

    if voice_ids:
        wanted = set(voice_ids)
        voices = [v for v in voices if v["voice_id"] in wanted]
    if category:
        voices = [v for v in voices if v["category"] == category]
    if search:
        voices = [v for v in voices if _matches_search(v, search)]

    if sort == "name":
        voices = sorted(voices, key=lambda v: v["name"])
    # ``created_at_unix`` is null for Moonshine voices; sort is a no-op there.
    if sort_direction == "desc":
        voices = list(reversed(voices))

    total_count = len(voices)
    start = 0
    if next_page_token:
        try:
            start = int(next_page_token)
        except ValueError:
            raise HTTPException(status_code=422, detail="invalid next_page_token")
    end = start + page_size
    page = voices[start:end]
    has_more = end < total_count

    body: dict[str, Any] = {
        "voices": [_public_voice(v) for v in page],
        "has_more": has_more,
        "next_page_token": str(end) if has_more else None,
    }
    if include_total_count:
        body["total_count"] = total_count
    return body


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------


class _VoiceSettings(BaseModel):
    speed: Optional[float] = None
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    style: Optional[float] = None
    use_speaker_boost: Optional[bool] = None


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    model_id: Optional[str] = None
    language_code: Optional[str] = None
    voice_settings: Optional[_VoiceSettings] = None


def handle_text_to_speech(
    voice_id: str,
    body: TextToSpeechRequest,
    output_format: Optional[str],
    default_language: str,
) -> Response:
    fmt, _sample_rate_hint = _parse_output_format(output_format)
    media_type = _FORMAT_MEDIA_TYPES[fmt]

    language = (body.language_code or default_language).replace("_", "-")
    resolved_voice = _resolve_voice(voice_id)
    speed = body.voice_settings.speed if body.voice_settings else None

    from bootlegger.speech import _get_tts

    try:
        tts = _get_tts(language, resolved_voice)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        kwargs: dict[str, Any] = {}
        if speed is not None:
            kwargs["speed"] = speed
        samples, sample_rate = tts.synthesize(body.text, **kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    audio_bytes = _encode(samples, sample_rate, fmt)
    return Response(content=audio_bytes, media_type=media_type)


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------

_ALLOWED_GRANULARITIES = {"none", "word"}


def _format_words(transcript: Any, granularity: str) -> list[dict[str, Any]]:
    if granularity == "none":
        return []

    out: list[dict[str, Any]] = []
    for line in transcript.lines:
        for word in line.words or []:
            out.append(
                {
                    "text": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "type": "word",
                    "logprob": float(word.confidence),
                }
            )
    return out


def handle_speech_to_text(
    transcriber: Any,
    lock: threading.Lock,
    file: Optional[UploadFile],
    model_id: str,
    language_code: Optional[str],
    timestamps_granularity: str,
    default_language: str,
) -> JSONResponse:
    if file is None:
        raise HTTPException(status_code=422, detail="file is required")
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id is required")
    if timestamps_granularity not in _ALLOWED_GRANULARITIES:
        raise HTTPException(
            status_code=422,
            detail=f"timestamps_granularity must be one of {sorted(_ALLOWED_GRANULARITIES)}",
        )

    file_bytes = file.file.read()
    filename = file.filename or "audio.wav"
    samples, sample_rate = decode_audio(file_bytes, filename)

    with lock:
        transcript = transcriber.transcribe_without_streaming(samples, sample_rate)

    full_text = " ".join(line.text for line in transcript.lines)
    duration = (
        transcript.lines[-1].start_time + transcript.lines[-1].duration if transcript.lines else 0.0
    )
    language = (language_code or default_language).split("-")[0]

    body = {
        "language_code": language,
        "language_probability": 1.0,
        "text": full_text,
        "words": _format_words(transcript, timestamps_granularity),
        "channel_index": None,
        "transcription_id": None,
        "audio_duration_secs": duration,
    }
    return JSONResponse(body)


# ---------------------------------------------------------------------------
# FastAPI route registration
# ---------------------------------------------------------------------------


def register_routes(app, settings) -> None:
    """Attach ElevenLabs-compatible routes to ``app`` using ``settings`` for defaults."""

    @app.get("/v2/voices")
    def get_voices(
        next_page_token: Optional[str] = None,
        page_size: int = 10,
        search: Optional[str] = None,
        sort: Optional[str] = None,
        sort_direction: Optional[str] = None,
        voice_type: Optional[str] = None,
        category: Optional[str] = None,
        include_total_count: bool = True,
    ):
        return list_voices(
            next_page_token=next_page_token,
            page_size=page_size,
            search=search,
            sort=sort,
            sort_direction=sort_direction,
            voice_type=voice_type,
            category=category,
            include_total_count=include_total_count,
        )

    @app.post("/v1/text-to-speech/{voice_id}")
    def text_to_speech(
        voice_id: str,
        body: TextToSpeechRequest,
        output_format: Optional[str] = None,
    ):
        return handle_text_to_speech(voice_id, body, output_format, settings.tts_language)

    @app.post("/v1/speech-to-text")
    def speech_to_text(
        model_id: str = Form(...),
        file: UploadFile = File(None),
        language_code: Optional[str] = Form(None),
        timestamps_granularity: str = Form("word"),
        tag_audio_events: bool = Form(True),
        diarize: bool = Form(False),
        num_speakers: Optional[int] = Form(None),
    ):
        return handle_speech_to_text(
            app.state.transcriber,
            app.state.lock,
            file,
            model_id,
            language_code,
            timestamps_granularity,
            settings.language,
        )
