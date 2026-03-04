# Bootlegger

An OpenAI-compatible speech-to-text API server powered by [Moonshine](https://github.com/moonshine-ai/moonshine). Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint that runs entirely on-device.

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/).

```bash
# Install ffmpeg (if not already installed)
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Install bootlegger
pip install .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install .
```

## Run

```bash
bootlegger
```

The server starts on `http://0.0.0.0:8000`. On first launch, Moonshine model weights are downloaded automatically.

### Configuration

All settings are controlled via environment variables with a `BOOTLEGGER_` prefix:

| Variable | Default | Description |
|---|---|---|
| `BOOTLEGGER_HOST` | `0.0.0.0` | Bind address |
| `BOOTLEGGER_PORT` | `8000` | Bind port |
| `BOOTLEGGER_LANGUAGE` | `en` | Transcription language |
| `BOOTLEGGER_MODEL_ARCH` | _(auto)_ | Moonshine model architecture (integer) |
| `BOOTLEGGER_API_PREFIX` | `/v1` | API route prefix |

Example:

```bash
BOOTLEGGER_PORT=9000 BOOTLEGGER_LANGUAGE=en bootlegger
```

## Docker

```bash
docker build -t bootlegger .
docker run -p 8000:8000 bootlegger
```

## API

### Transcribe audio

```
POST /v1/audio/transcriptions
```

Multipart form fields:

| Field | Required | Default | Description |
|---|---|---|---|
| `file` | yes | | Audio file (wav, mp3, ogg, flac, m4a, webm) |
| `model` | no | `moonshine` | Model name (ignored, for OpenAI compatibility) |
| `language` | no | server default | Language code |
| `response_format` | no | `json` | One of: `json`, `text`, `verbose_json`, `srt`, `vtt` |

Example with curl:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F response_format=json
```

Response (`json`):

```json
{"text": "transcribed text here"}
```

Response (`verbose_json`):

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 3.5,
  "text": "transcribed text here",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "transcribed text here"
    }
  ]
}
```

### List models

```
GET /v1/models
```

```bash
curl http://localhost:8000/v1/models
```

### OpenAI SDK

Bootlegger works with the standard OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

with open("recording.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(model="moonshine", file=f)
    print(transcript.text)
```
