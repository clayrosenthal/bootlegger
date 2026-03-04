from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "BOOTLEGGER_"}

    host: str = "0.0.0.0"
    port: int = 8000
    language: str = "en"
    model_arch: int | None = None
    api_prefix: str = "/v1"
