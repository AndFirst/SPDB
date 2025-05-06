from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict

env_file = ".env"


class Environment(StrEnum):
    development = "development"
    production = "production"


class Settings(BaseSettings):
    ENV: Environment = Environment.development
    DEBUG: bool = False
    LOG_LEVEL: str = "DEBUG"
    SECRET: str = "secret"
    CORS_ORIGINS: list[str] = []
    ALLOWED_HOSTS: set[str] = ["127.0.0.1:3000", "localhost:3000"]
    BASE_URL: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_file=env_file,
        extra="allow",
    )


settings = Settings()
