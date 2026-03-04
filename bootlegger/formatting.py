from moonshine_voice import Transcript


def _full_text(transcript: Transcript) -> str:
    return " ".join(line.text for line in transcript.lines)


def _total_duration(transcript: Transcript) -> float:
    if not transcript.lines:
        return 0.0
    last = transcript.lines[-1]
    return last.start_time + last.duration


def _format_ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_json(transcript: Transcript) -> dict:
    return {"text": _full_text(transcript)}


def format_text(transcript: Transcript) -> str:
    return _full_text(transcript)


def format_verbose_json(transcript: Transcript, language: str) -> dict:
    segments = []
    for i, line in enumerate(transcript.lines):
        segments.append(
            {
                "id": i,
                "seek": 0,
                "start": line.start_time,
                "end": line.start_time + line.duration,
                "text": line.text,
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
            }
        )
    return {
        "task": "transcribe",
        "language": language,
        "duration": _total_duration(transcript),
        "text": _full_text(transcript),
        "segments": segments,
    }


def format_srt(transcript: Transcript) -> str:
    blocks = []
    for i, line in enumerate(transcript.lines, 1):
        start = _format_ts_srt(line.start_time)
        end = _format_ts_srt(line.start_time + line.duration)
        blocks.append(f"{i}\n{start} --> {end}\n{line.text}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def format_vtt(transcript: Transcript) -> str:
    blocks = ["WEBVTT", ""]
    for line in transcript.lines:
        start = _format_ts_vtt(line.start_time)
        end = _format_ts_vtt(line.start_time + line.duration)
        blocks.append(f"{start} --> {end}\n{line.text}")
    return "\n\n".join(blocks) + "\n" if len(blocks) > 2 else "WEBVTT\n"
