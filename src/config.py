from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required
    openai_api_key: str
    api_token: str

    # OpenAI
    openai_base_url: str | None = None  # Custom OpenAI API base URL

    # Service
    host: str = "0.0.0.0"
    port: int = 8000
    # Internal admapix API key for website-initiated research
    internal_api_key: str | None = None

    # Paths
    storage_path: Path = Path("./output")
    projects_path: Path = Path("./projects")

    # Defaults
    log_level: str = "info"
    default_model: str = "o3-deep-research"
    max_concurrent_tasks: int = 5
    output_retention_days: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings: Settings | None = None


def get_settings() -> Settings:
    global settings
    if settings is None:
        settings = Settings()
    return settings
