from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
_APP_DIR = Path(__file__).resolve().parent


def load_dotenv_files() -> None:
    """Load environment variables from root `.env` (preferred) or `app/.env` if present."""
    for env_path in (_ROOT_DIR / ".env", _APP_DIR / ".env"):
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


load_dotenv_files()


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5432/wismo",
        )
    )
    phone_hmac_secret: str = field(
        default_factory=lambda: os.getenv(
            "PHONE_HMAC_SECRET",
            "wismo-dev-hmac-secret-key",
        )
    )
    default_phone_region: str = field(
        default_factory=lambda: os.getenv("DEFAULT_PHONE_REGION", "IN")
    )
    llm_model: str = field(default="gpt-6-luna")
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "llm_model", "gpt-6-luna")


def get_settings() -> Settings:
    return Settings()
