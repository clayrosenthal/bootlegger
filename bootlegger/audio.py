import io
import struct

from pydub import AudioSegment


def decode_audio(file_bytes: bytes, filename: str) -> tuple[list[float], int]:
    """Decode audio bytes to mono PCM float samples and sample rate."""
    buf = io.BytesIO(file_bytes)

    # Determine format from extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else None
    format_map = {
        "wav": "wav",
        "mp3": "mp3",
        "mpga": "mp3",
        "ogg": "ogg",
        "flac": "flac",
        "m4a": "m4a",
        "webm": "webm",
    }
    fmt = format_map.get(ext)

    if fmt:
        audio = AudioSegment.from_file(buf, format=fmt)
    else:
        audio = AudioSegment.from_file(buf)

    # Convert to mono
    if audio.channels > 1:
        audio = audio.set_channels(1)

    sample_rate = audio.frame_rate
    sample_width = audio.sample_width

    # Extract raw PCM and convert to float [-1.0, 1.0]
    raw = audio.raw_data
    if sample_width == 2:
        fmt_char = "<h"
        max_val = 32768.0
    elif sample_width == 4:
        fmt_char = "<i"
        max_val = 2147483648.0
    elif sample_width == 1:
        # 8-bit PCM is unsigned
        samples = [((b - 128) / 128.0) for b in raw]
        return samples, sample_rate
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    n_samples = len(raw) // sample_width
    samples = [
        struct.unpack_from(fmt_char, raw, i * sample_width)[0] / max_val
        for i in range(n_samples)
    ]
    return samples, sample_rate
