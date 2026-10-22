from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_secret_file(secret_path: str) -> str:
    """Read a secret from a file (Docker secrets support)."""
    try:
        return Path(secret_path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _with_psycopg3_driver(url: str) -> str:
    """Pin the psycopg3 driver so a bare ``postgresql://`` never resolves to psycopg2.

    Requirements install ``psycopg[binary]`` (psycopg 3), but SQLAlchemy's bare
    ``postgresql://`` defaults to psycopg2, which is not installed — the engine
    would fail at import. Explicitly selecting ``postgresql+psycopg`` avoids that.
    """
    prefix = "postgresql://"
    if url.startswith(prefix):
        return f"postgresql+psycopg://{url[len(prefix):]}"
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "NEPS ML/AI Service"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"

    # Database
    DATABASE_URL: str = "postgresql+psycopg://neps:neps_password@localhost:5432/neps_db"
    DATABASE_URL_FILE: str | None = None
    DATABASE_SSL_MODE: str = "require"

    @property
    def database_url(self) -> str:
        """Get database URL from file or environment, always on the psycopg3 driver."""
        if self.DATABASE_URL_FILE:
            password = _read_secret_file(self.DATABASE_URL_FILE)
            # Construct URL with password from secret
            return _with_psycopg3_driver(
                f"postgresql://neps:{password}@postgres:5432/neps_db"
                f"?sslmode={self.DATABASE_SSL_MODE}"
            )
        return _with_psycopg3_driver(self.DATABASE_URL)


@lru_cache
def get_settings() -> Settings:
    return Settings()
