import threading

from fastapi import UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

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
):
    file_bytes = file.file.read()
    filename = file.filename or "audio.wav"

    samples, sample_rate = decode_audio(file_bytes, filename)

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
