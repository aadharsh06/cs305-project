
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    server_url: str = ""
    api_token: str = ""
    request_timeout_seconds: float = 900

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_prefix="RAG_", extra="ignore"
    )

def get_settings() -> Settings:
    return Settings()
