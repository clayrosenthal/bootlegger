import io
import re
import struct
import subprocess
import threading
import wave
from queue import Queue
from typing import Iterator, Optional, Union

from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
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

_FFMPEG_OUTPUT_ARGS = {
    "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"],
    "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
    "flac": ["-f", "ogg", "-c:a", "flac"],
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


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()] or [text.strip()]


def _streaming_wav_header(sample_rate: int) -> bytes:
    # RIFF header with 0xFFFFFFFF placeholders (streaming WAV convention).
    chunk_size = 0xFFFFFFFF
    data_size = 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )


def _synthesize_sentences(tts, sentences: list[str], speed: Optional[float]) -> Queue:
    """Spawn a producer thread that synthesizes each sentence and puts
    ('data', sample_rate, pcm_bytes) or ('error', exc) or ('eof', None) on a queue."""
    queue: Queue = Queue(maxsize=2)

    def produce():
        try:
            for sent in sentences:
                kwargs = {}
                if speed is not None:
                    kwargs["speed"] = speed
                samples, sr = tts.synthesize(sent, **kwargs)
                queue.put(("data", sr, _samples_to_int16(samples)))
        except Exception as exc:
            queue.put(("error", exc, None))
        finally:
            queue.put(("eof", None, None))

    threading.Thread(target=produce, daemon=True).start()
    return queue


def _stream_raw_pcm(queue: Queue) -> Iterator[bytes]:
    while True:
        kind, a, b = queue.get()
        if kind == "eof":
            return
        if kind == "error":
            raise a
        yield b


def _stream_wav(queue: Queue) -> Iterator[bytes]:
    header_sent = False
    while True:
        kind, a, b = queue.get()
        if kind == "eof":
            return
        if kind == "error":
            raise a
        if not header_sent:
            yield _streaming_wav_header(a)
            header_sent = True
        yield b


def _stream_via_ffmpeg(queue: Queue, fmt: str) -> Iterator[bytes]:
    first = queue.get()
    kind, sr_or_exc, pcm = first
    if kind == "eof":
        return
    if kind == "error":
        raise sr_or_exc
    sample_rate = sr_or_exc

    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *_FFMPEG_OUTPUT_ARGS[fmt],
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    writer_error: list[BaseException] = []

    def write_pcm():
        try:
            assert proc.stdin is not None
            proc.stdin.write(pcm)
            while True:
                kind2, a2, b2 = queue.get()
                if kind2 == "eof":
                    break
                if kind2 == "error":
                    writer_error.append(a2)
                    break
                proc.stdin.write(b2)
        except BrokenPipeError:
            pass
        except Exception as exc:
            writer_error.append(exc)
        finally:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass

    writer = threading.Thread(target=write_pcm, daemon=True)
    writer.start()
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
    finally:
        writer.join(timeout=5)
        proc.wait(timeout=5)
        if writer_error:
            raise writer_error[0]
        if proc.returncode not in (0, None):
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {stderr}")


def handle_speech(req: SpeechRequest, default_language: str) -> Response:
    fmt = (req.response_format or "mp3").lower()
    if fmt not in _FORMAT_MEDIA_TYPES:
        raise HTTPException(
            status_code=400, detail=f"Unsupported response_format: {fmt}"
        )

    language = (req.language or default_language).replace("_", "-")
    voice = _resolve_voice(req.voice)

    if req.stream_format is not None and req.stream_format != "audio":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported stream_format: {req.stream_format} (only 'audio' is supported)",
        )

    streaming = req.stream_format == "audio"

    try:
        tts = _get_tts(language, voice)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    media_type = _FORMAT_MEDIA_TYPES[fmt]

    if streaming:
        sentences = _split_sentences(req.input)
        queue = _synthesize_sentences(tts, sentences, req.speed)
        if fmt == "pcm":
            iterator = _stream_raw_pcm(queue)
        elif fmt == "wav":
            iterator = _stream_wav(queue)
        else:
            iterator = _stream_via_ffmpeg(queue, fmt)
        return StreamingResponse(iterator, media_type=media_type)

    try:
        kwargs = {}
        if req.speed is not None:
            kwargs["speed"] = req.speed
        samples, sample_rate = tts.synthesize(req.input, **kwargs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    audio_bytes = _encode(samples, sample_rate, fmt)
    return Response(content=audio_bytes, media_type=media_type)
