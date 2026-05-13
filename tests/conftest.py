import io
import struct
import wave
from dataclasses import dataclass, field
from typing import Any

import pytest


def _pcm_int16(samples: list[float]) -> bytes:
    out = bytearray(len(samples) * 2)
    for i, s in enumerate(samples):
        if s > 1.0:
            s = 1.0
        elif s < -1.0:
            s = -1.0
        struct.pack_into("<h", out, i * 2, int(s * 32767.0))
    return bytes(out)


def make_wav_bytes(
    samples: list[float] | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    if samples is None:
        samples = [0.0] * sample_rate  # 1 second of silence
    pcm = _pcm_int16(samples)
    if channels > 1:
        # Duplicate mono frames into N channels.
        frames = bytearray()
        for i in range(0, len(pcm), 2):
            frames.extend(pcm[i : i + 2] * channels)
        pcm = bytes(frames)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav_bytes()


@dataclass
class FakeLine:
    text: str
    start_time: float
    duration: float


@dataclass
class FakeTranscript:
    lines: list[FakeLine] = field(default_factory=list)


@pytest.fixture
def fake_transcript() -> FakeTranscript:
    return FakeTranscript(
        lines=[
            FakeLine(text="hello world", start_time=0.0, duration=1.5),
            FakeLine(text="goodbye", start_time=1.5, duration=2.0),
        ]
    )


class FakeTranscriber:
    def __init__(self, transcript: Any):
        self._transcript = transcript

    def transcribe_without_streaming(self, samples, sample_rate):
        return self._transcript


@pytest.fixture
def fake_transcriber(fake_transcript) -> FakeTranscriber:
    return FakeTranscriber(fake_transcript)
