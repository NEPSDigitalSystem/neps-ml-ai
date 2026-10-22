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

    # Database — mirrors neps-backend: the URL is always assembled in code from
    # these parts, so a Docker Secret file (Docker) or an explicit value (local
    # development) is the only way to supply credentials. There is deliberately
    # no ``DATABASE_URL`` override: a stale ``.env`` must not be able to shadow
    # the mounted secrets.
    DATABASE_HOST: str = "postgres"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "neps_db"
    DATABASE_USER: str | None = None
    DATABASE_PASSWORD: str | None = None
    DATABASE_USER_FILE: str | None = None
    DATABASE_PASSWORD_FILE: str | None = None
    DATABASE_SSL_MODE: str | None = "require"

    @property
    def database_user(self) -> str:
        if self.DATABASE_USER_FILE:
            secret = _read_secret_file(self.DATABASE_USER_FILE)
            if secret:
                return secret
        return self.DATABASE_USER or "neps"

    @property
    def database_password(self) -> str:
        if self.DATABASE_PASSWORD_FILE:
            secret = _read_secret_file(self.DATABASE_PASSWORD_FILE)
            if secret:
                return secret
        return self.DATABASE_PASSWORD or "neps_password"

    @property
    def database_url(self) -> str:
        """Assemble the URL from parts — same contract as ``neps-backend``.

        Host/port/name come from the environment; user and password come from a
        mounted Docker Secret file when present, otherwise from the plain
        ``DATABASE_USER`` / ``DATABASE_PASSWORD`` values used for local
        development. No credential is hardcoded and the username is never
        assumed, so this cannot drift from the infra ``postgres_user`` secret.
        """
        ssl_query = f"?sslmode={self.DATABASE_SSL_MODE}" if self.DATABASE_SSL_MODE else ""
        return _with_psycopg3_driver(
            f"postgresql://{self.database_user}:{self.database_password}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}{ssl_query}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
