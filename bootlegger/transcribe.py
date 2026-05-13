import json
import threading
from queue import Queue
from typing import Iterator

from fastapi import UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from bootlegger.audio import decode_audio
from bootlegger.formatting import (
    format_json,
    format_text,
    format_verbose_json,
    format_srt,
    format_vtt,
)


def handle_transcription(
    transcriber,
    lock: threading.Lock,
    file: UploadFile,
    language: str,
    response_format: str,
    stream: bool = False,
):
    file_bytes = file.file.read()
    filename = file.filename or "audio.wav"

    samples, sample_rate = decode_audio(file_bytes, filename)

    if stream:
        return StreamingResponse(
            _stream_sse(transcriber, lock, samples, sample_rate, language),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    with lock:
        transcript = transcriber.transcribe_without_streaming(samples, sample_rate)

    if response_format == "text":
        return PlainTextResponse(format_text(transcript))
    elif response_format == "verbose_json":
        return JSONResponse(format_verbose_json(transcript, language))
    elif response_format == "srt":
        return PlainTextResponse(format_srt(transcript), media_type="text/plain")
    elif response_format == "vtt":
        return PlainTextResponse(format_vtt(transcript), media_type="text/vtt")
    else:  # json
        return JSONResponse(format_json(transcript))


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_sse(
    transcriber,
    lock: threading.Lock,
    samples,
    sample_rate: int,
    language: str,
) -> Iterator[str]:
    from moonshine_voice import TranscriptEventListener

    queue: Queue = Queue()

    class _Listener(TranscriptEventListener):
        def on_line_completed(self, event):
            line = event.line
            queue.put(("complete", line.line_id, line.text, line))

    def produce():
        try:
            with lock:
                listener = _Listener()
                transcriber.add_listener(listener)
                transcriber.start()
                try:
                    chunk_size = max(1, int(0.1 * sample_rate))
                    for i in range(0, len(samples), chunk_size):
                        transcriber.add_audio(
                            samples[i:i + chunk_size], sample_rate
                        )
                finally:
                    transcriber.stop()
                    transcriber.remove_listener(listener)
        except Exception as exc:
            queue.put(("error", None, None, exc))
        finally:
            queue.put(("done", None, None, None))

    threading.Thread(target=produce, daemon=True).start()

    completed_lines: list = []

    while True:
        kind, lid, text, payload = queue.get()
        if kind == "done":
            break
        if kind == "error":
            yield _sse({"type": "error", "error": str(payload)})
            return

        if kind == "complete":
            prefix = " " if completed_lines else ""
            yield _sse({"type": "transcript.text.delta", "delta": prefix + text})
            completed_lines.append(text)

    yield _sse({"type": "transcript.text.done", "text": " ".join(completed_lines)})
    yield "data: [DONE]\n\n"
