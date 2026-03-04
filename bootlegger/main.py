import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

from bootlegger.config import Settings
from bootlegger.transcribe import handle_transcription

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from moonshine_voice import Transcriber, ModelArch, get_model_for_language

    model_arch = ModelArch(settings.model_arch) if settings.model_arch is not None else None
    model_path, resolved_arch = get_model_for_language(settings.language, model_arch)

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


app = FastAPI(title="Bootlegger", version="0.1.0", lifespan=lifespan)


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


def cli():
    uvicorn.run(
        "bootlegger.main:app",
        host=settings.host,
        port=settings.port,
    )


if __name__ == "__main__":
    cli()
