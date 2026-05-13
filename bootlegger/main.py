#!/usr/bin/env python3

import argparse
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from bootlegger import __version__
from bootlegger.config import Settings
from bootlegger.speech import SpeechRequest, handle_speech
from bootlegger.transcribe import handle_transcription


def create_app(settings: Settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from moonshine_voice import Transcriber, ModelArch, get_model_for_language

        model_arch = (
            ModelArch(settings.model_arch)
            if settings.model_arch is not None
            else None
        )
        model_path, resolved_arch = get_model_for_language(
            settings.language, model_arch
        )

        transcriber = Transcriber(model_path, resolved_arch)
        app.state.transcriber = transcriber
        app.state.lock = threading.Lock()
        app.state.model_path = model_path
        app.state.model_arch = resolved_arch
        app.state.language = settings.language
        try:
            yield
        finally:
            transcriber.close()

    app = FastAPI(title="Bootlegger", version=__version__, lifespan=lifespan)

    @app.post(settings.api_prefix + "/audio/transcriptions")
    def transcribe(
        file: UploadFile = File(...),
        model: str = Form("moonshine"),
        language: str | None = Form(None),
        response_format: str = Form("json"),
        prompt: str | None = Form(None),
        temperature: float | None = Form(None),
    ):
        lang = language or app.state.language
        return handle_transcription(
            app.state.transcriber,
            app.state.lock,
            file,
            lang,
            response_format,
        )

    @app.post(settings.api_prefix + "/audio/speech")
    def speech(req: SpeechRequest):
        return handle_speech(req, settings.tts_language)

    @app.get(settings.api_prefix + "/models")
    def list_models():
        from moonshine_voice import model_arch_to_string

        arch_str = model_arch_to_string(app.state.model_arch)
        model_id = f"moonshine-{arch_str}"
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "moonshine",
                }
            ],
        }

    return app


settings = Settings()
app = create_app(settings)


def _build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="bootlegger",
        description="OpenAI-compatible STT/TTS server backed by Moonshine.",
    )
    parser.add_argument("--version", "-V", action="version", version=f"bootlegger {__version__}")
    parser.add_argument("--host", default=None, help=f"Bind address (default: {defaults.host})")
    parser.add_argument("--port", type=int, default=None, help=f"Bind port (default: {defaults.port})")
    parser.add_argument("--language", default=None, help=f"Default STT language (default: {defaults.language})")
    parser.add_argument("--model-arch", type=int, default=None, help="Moonshine STT model architecture (integer)")
    parser.add_argument("--api-prefix", default=None, help=f"API route prefix (default: {defaults.api_prefix})")
    parser.add_argument("--tts-language", default=None, help=f"Default TTS language tag (default: {defaults.tts_language})")
    parser.add_argument("--tts-voice", default=None, help="Default TTS voice id (e.g. kokoro_af_alloy)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (development only)")
    parser.add_argument("--log-level", default=None, help="Uvicorn log level (debug, info, warning, error, critical)")
    return parser


def cli():
    global settings, app

    parser = _build_parser()
    args = parser.parse_args()

    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in {"reload", "log_level"}
    }
    if overrides:
        settings = settings.model_copy(update=overrides)
        app = create_app(settings)

    uvicorn_kwargs = {"host": settings.host, "port": settings.port}
    if args.log_level is not None:
        uvicorn_kwargs["log_level"] = args.log_level
    if args.reload:
        uvicorn.run("bootlegger.main:app", reload=True, **uvicorn_kwargs)
    else:
        uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    cli()
