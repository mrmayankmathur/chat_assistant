from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-opus-5-5")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    )


def get_settings() -> Settings:
    return Settings()
