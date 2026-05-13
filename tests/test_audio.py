import math

import pytest

from bootlegger.audio import decode_audio
from tests.conftest import make_wav_bytes


def test_decode_wav_roundtrip_silence():
    wav = make_wav_bytes(samples=[0.0] * 8000, sample_rate=16000)
    samples, sr = decode_audio(wav, "audio.wav")
    assert sr == 16000
    assert len(samples) == 8000
    assert all(s == 0.0 for s in samples)


def test_decode_wav_with_signal_preserves_amplitude():
    sr = 16000
    sig = [math.sin(2 * math.pi * 440 * t / sr) * 0.5 for t in range(sr // 4)]
    wav = make_wav_bytes(samples=sig, sample_rate=sr)
    samples, decoded_sr = decode_audio(wav, "tone.wav")
    assert decoded_sr == sr
    assert len(samples) == len(sig)
    # 16-bit quantization tolerance.
    for original, decoded in zip(sig, samples):
        assert abs(original - decoded) < 1e-3


def test_decode_wav_stereo_is_mixed_to_mono():
    wav = make_wav_bytes(samples=[0.0] * 4000, sample_rate=8000, channels=2)
    samples, sr = decode_audio(wav, "stereo.wav")
    assert sr == 8000
    # After mono mixdown, sample count equals frame count.
    assert len(samples) == 4000


def test_decode_unknown_extension_falls_back_to_format_detection():
    wav = make_wav_bytes(samples=[0.0] * 1600, sample_rate=16000)
    samples, sr = decode_audio(wav, "no_extension")
    assert sr == 16000
    assert len(samples) == 1600


def test_decode_invalid_bytes_raises():
    with pytest.raises(Exception):
        decode_audio(b"not really audio", "broken.wav")
