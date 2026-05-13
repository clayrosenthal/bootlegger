# Bootlegger

An OpenAI-compatible audio API server powered by [Moonshine](https://github.com/moonshine-ai/moonshine). Drop-in replacement for OpenAI's `/v1/audio/transcriptions` and `/audio/speech` endpoints that runs entirely on-device.

## Install

Requires Python 3.10+ and [ffmpeg](https://ffmpeg.org/).

```bash
# Install ffmpeg (if not already installed)
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Install bootlegger
pip install bootlegger-voice
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install bootlegger-voice
```

## Run

```bash
bootlegger
```

The server starts on `http://0.0.0.0:8000`. On first launch, Moonshine model weights are downloaded automatically.

### Configuration

Settings can be supplied via CLI flags or environment variables (CLI flags win). Run `bootlegger --help` for the full list:

```
bootlegger --port 9000 --tts-voice kokoro_af_alloy --tts-language en-us
```

Environment variables use the `BOOTLEGGER_` prefix:

| Variable | Default | Description |
|---|---|---|
| `BOOTLEGGER_HOST` | `0.0.0.0` | Bind address |
| `BOOTLEGGER_PORT` | `8000` | Bind port |
| `BOOTLEGGER_LANGUAGE` | `en` | Transcription language |
| `BOOTLEGGER_MODEL_ARCH` | _(auto)_ | Moonshine model architecture (integer) |
| `BOOTLEGGER_API_PREFIX` | `/v1` | API route prefix |
| `BOOTLEGGER_TTS_LANGUAGE` | `en-us` | Default TTS language tag (e.g. `en-us`, `es-es`, `ja-jp`) |
| `BOOTLEGGER_TTS_VOICE` | _(none)_ | Default TTS voice id (e.g. `kokoro_af_alloy`) |

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
| `response_format` | no | `json` | One of: `json`, `text`, `verbose_json`, `srt`, `vtt` (ignored when `stream=true`) |
| `stream` | no | `false` | When `true`, returns a `text/event-stream` of OpenAI-shaped `transcript.text.delta` events followed by `transcript.text.done` and `[DONE]`. Each delta corresponds to a completed VAD-detected line, so concatenating deltas reconstructs the final text exactly. |

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

### Generate speech (TTS)

```
POST /v1/audio/speech
```

Synthesizes audio on-device using [Moonshine's TextToSpeech](https://github.com/moonshine-ai/moonshine) (Kokoro and Piper voices). On first use, Moonshine downloads the required voice + G2P models to its cache; subsequent calls run fully offline.

JSON body fields:

| Field | Required | Default | Description |
|---|---|---|---|
| `input` | yes | | Text to synthesize (max 4096 chars) |
| `model` | yes | | Ignored; for OpenAI compatibility |
| `voice` | no | `alloy` | OpenAI alias (`alloy`, `echo`, `nova`, `onyx`), a Moonshine id like `kokoro_af_alloy`, or `{"id": "voice_..."}` |
| `response_format` | no | `mp3` | One of: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | no | `1.0` | Speaking-rate multiplier |
| `language` | no | server default | TTS language tag, e.g. `en-us`, `es-es`, `ja-jp`. See `list_tts_languages()` for the full set |
| `stream_format` | no | _(none)_ | Set to `"audio"` to stream audio as it's synthesized. Input is split into sentences; each is synthesized in a background thread while earlier bytes are flushed to the client. SSE is not supported. |

Encoded formats other than `wav`/`pcm` use `ffmpeg` via `pydub`, so make sure `ffmpeg` is installed.

The response body is the raw audio bytes; the `Content-Type` matches the requested format.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world","model":"tts-1","voice":"alloy"}' \
  --output speech.mp3
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with client.audio.speech.with_streaming_response.create(
    model="tts-1", voice="alloy", input="Hello world"
) as r:
    r.stream_to_file("speech.mp3")
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
