"""Application configuration.

All tunables live here and are sourced from environment variables or a local
`.env` file. Nothing else in the codebase should read `os.environ` directly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent

DevicePreference = Literal["auto", "cuda", "cpu"]
LogFormat = Literal["text", "json"]


class Settings(BaseSettings):
    """Runtime settings, immutable for the lifetime of the process."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # -- application ---------------------------------------------------
    app_name: str = "PixelForge AI"
    environment: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: LogFormat = "text"

    # -- http ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    # NoDecode: pydantic-settings would otherwise JSON-decode list fields inside
    # the dotenv source, before validators run, so a plain CSV value in .env
    # would raise instead of reaching _split_csv below.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # -- storage -------------------------------------------------------
    storage_dir: Path = REPO_ROOT / "storage"
    models_dir: Path = REPO_ROOT / "models"
    # Left unset by default and derived from storage_dir below. A literal
    # relative URL would resolve against the process working directory,
    # so the database would move depending on where uvicorn was launched.
    database_url: str | None = None
    # Apply migrations at startup. Convenient locally; production should
    # run "alembic upgrade head" as a deploy step and set this false.
    auto_migrate: bool = True

    # -- upload limits (security § 16) ---------------------------------
    max_upload_size_mb: int = Field(default=32, ge=1, le=512)
    max_input_pixels: int = Field(default=16_000_000, ge=1_000)
    min_input_dimension: int = Field(default=32, ge=1)
    max_output_pixels: int = Field(default=200_000_000, ge=1_000)
    allowed_formats: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["JPEG", "PNG", "WEBP"]
    )

    # -- inference -----------------------------------------------------
    device: DevicePreference = "auto"
    default_model: str = "RealESRGAN_x4plus"
    default_scale: int = 4
    tile_size: int = Field(default=256, ge=0, le=2048)
    tile_pad: int = Field(default=16, ge=0, le=128)
    use_fp16: bool = True
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)

    # -- retention -----------------------------------------------------
    temp_retention_hours: int = Field(default=24, ge=1)
    max_total_storage_gb: float = Field(default=10.0, gt=0)
    cleanup_interval_minutes: int = Field(default=30, ge=1)

    @field_validator("storage_dir", "models_dir", mode="after")
    @classmethod
    def _resolve_against_repo_root(cls, value: Path) -> Path:
        """Make directory settings absolute.

        A relative value such as "./storage" in .env would otherwise resolve
        against the process working directory, so the storage tree and the
        database would move depending on where uvicorn was started.
        """
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @field_validator("cors_origins", "allowed_formats", mode="before")
    @classmethod
    def _parse_list(cls, value: object) -> object:
        """Accept either a JSON array or a comma-separated string.

        These fields are annotated NoDecode, so pydantic-settings hands us the
        raw string from .env. CSV is what .env.example documents because it is
        far more readable; JSON is still accepted because it is the form
        pydantic-settings uses natively and people reasonably expect it.
        """
        if not isinstance(value, str):
            return value

        text = value.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Value looks like JSON but could not be parsed: {exc}") from exc
        return [item.strip() for item in text.split(",") if item.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def resolved_database_url(self) -> str:
        """Configured URL, or an absolute SQLite path inside the storage tree."""
        if self.database_url is not None:
            return self.database_url
        return f"sqlite+aiosqlite:///{(self.storage_dir / 'pixelforge.db').as_posix()}"

    @property
    def inputs_dir(self) -> Path:
        return self.storage_dir / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.storage_dir / "outputs"

    @property
    def thumbs_dir(self) -> Path:
        return self.storage_dir / "thumbs"

    def ensure_directories(self) -> None:
        """Create the storage tree. Safe to call repeatedly."""
        for path in (self.inputs_dir, self.outputs_dir, self.thumbs_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (also the FastAPI dependency)."""
    return Settings()
