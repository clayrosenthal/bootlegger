from bootlegger.formatting import (
    _format_ts_srt,
    _format_ts_vtt,
    _full_text,
    _total_duration,
    format_json,
    format_srt,
    format_text,
    format_verbose_json,
    format_vtt,
)
from tests.conftest import FakeLine, FakeTranscript


def test_full_text_joins_lines_with_space():
    t = FakeTranscript(lines=[FakeLine("a", 0, 1), FakeLine("b", 1, 1), FakeLine("c", 2, 1)])
    assert _full_text(t) == "a b c"


def test_total_duration_uses_last_line():
    t = FakeTranscript(lines=[FakeLine("a", 0, 0.5), FakeLine("b", 0.5, 1.25)])
    assert _total_duration(t) == 1.75


def test_total_duration_empty_transcript_is_zero():
    assert _total_duration(FakeTranscript(lines=[])) == 0.0


def test_format_ts_srt_uses_comma_separator():
    assert _format_ts_srt(0.0) == "00:00:00,000"
    assert _format_ts_srt(3661.5) == "01:01:01,500"


def test_format_ts_vtt_uses_dot_separator():
    assert _format_ts_vtt(0.0) == "00:00:00.000"
    assert _format_ts_vtt(3661.5) == "01:01:01.500"


def test_format_json_returns_text(fake_transcript):
    assert format_json(fake_transcript) == {"text": "hello world goodbye"}


def test_format_text_returns_plain_string(fake_transcript):
    assert format_text(fake_transcript) == "hello world goodbye"


def test_format_verbose_json_shape(fake_transcript):
    result = format_verbose_json(fake_transcript, "en")
    assert result["task"] == "transcribe"
    assert result["language"] == "en"
    assert result["text"] == "hello world goodbye"
    assert result["duration"] == 3.5
    assert len(result["segments"]) == 2

    seg0 = result["segments"][0]
    assert seg0["id"] == 0
    assert seg0["start"] == 0.0
    assert seg0["end"] == 1.5
    assert seg0["text"] == "hello world"


def test_format_srt_blocks(fake_transcript):
    out = format_srt(fake_transcript)
    assert "1\n00:00:00,000 --> 00:00:01,500\nhello world" in out
    assert "2\n00:00:01,500 --> 00:00:03,500\ngoodbye" in out
    # Trailing newline after each block separation.
    assert out.endswith("\n")


def test_format_srt_empty():
    assert format_srt(FakeTranscript(lines=[])) == ""


def test_format_vtt_starts_with_header(fake_transcript):
    out = format_vtt(fake_transcript)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500\nhello world" in out
    assert "00:00:01.500 --> 00:00:03.500\ngoodbye" in out


def test_format_vtt_empty_returns_header_only():
    assert format_vtt(FakeTranscript(lines=[])) == "WEBVTT\n"
