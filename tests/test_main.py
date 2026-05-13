import pytest

from bootlegger.main import _build_parser


def test_parser_help_lists_all_flags():
    parser = _build_parser()
    help_text = parser.format_help()
    for flag in [
        "--host",
        "--port",
        "--language",
        "--model-arch",
        "--api-prefix",
        "--tts-language",
        "--tts-voice",
        "--reload",
        "--log-level",
        "--version",
    ]:
        assert flag in help_text


def test_parser_defaults_are_none_for_overrides():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.host is None
    assert args.port is None
    assert args.language is None
    assert args.model_arch is None
    assert args.api_prefix is None
    assert args.tts_language is None
    assert args.tts_voice is None
    assert args.reload is False
    assert args.log_level is None


def test_parser_accepts_overrides():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--language",
            "fr",
            "--model-arch",
            "1",
            "--api-prefix",
            "/api",
            "--tts-language",
            "es-es",
            "--tts-voice",
            "kokoro_af_nova",
            "--reload",
            "--log-level",
            "debug",
        ]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 9000
    assert args.language == "fr"
    assert args.model_arch == 1
    assert args.api_prefix == "/api"
    assert args.tts_language == "es-es"
    assert args.tts_voice == "kokoro_af_nova"
    assert args.reload is True
    assert args.log_level == "debug"


def test_parser_rejects_non_integer_port():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--port", "not-a-number"])


def test_parser_version_exits_cleanly():
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_create_app_registers_expected_routes():
    from bootlegger.config import Settings
    from bootlegger.main import create_app

    app = create_app(Settings(api_prefix="/v1"))
    paths = {route.path for route in app.routes}
    assert "/v1/audio/transcriptions" in paths
    assert "/v1/audio/speech" in paths
    assert "/v1/models" in paths


def test_create_app_honors_custom_api_prefix():
    from bootlegger.config import Settings
    from bootlegger.main import create_app

    app = create_app(Settings(api_prefix="/api/v2"))
    paths = {route.path for route in app.routes}
    assert "/api/v2/audio/transcriptions" in paths
    assert "/api/v2/audio/speech" in paths
    assert "/api/v2/models" in paths
