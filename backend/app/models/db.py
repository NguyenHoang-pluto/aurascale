"""SQLAlchemy ORM models.

Files live on disk; this table stores metadata and paths only. Path columns are
server-side detail and are never serialised into an API response (§ 16).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

from app.models.enums import JobStage, JobStatus


class Base(DeclarativeBase):
    """Declarative base. Alembic autogenerate targets this metadata."""


def utcnow() -> datetime:
    """Timezone-aware UTC now. Naive datetimes are a recurring source of
    off-by-hours bugs once a second timezone is involved."""
    return datetime.now(UTC)


class Job(Base):
    __tablename__ = "jobs"

    # -- identity ------------------------------------------------------
    # UUID4 hex. Also the on-disk filename stem, so it must never come from
    # user input (§ 16: generated filenames, no path traversal).
    id: Mapped[str] = mapped_column(String(32), primary_key=True)

    # -- state ---------------------------------------------------------
    status: Mapped[JobStatus] = mapped_column(String(16), default=JobStatus.QUEUED)
    stage: Mapped[JobStage | None] = mapped_column(String(24), default=None)
    progress: Mapped[int] = mapped_column(Integer, default=0)

    # -- request parameters --------------------------------------------
    model_name: Mapped[str] = mapped_column(String(64))
    scale: Mapped[int] = mapped_column(Integer)
    output_format: Mapped[str] = mapped_column(String(8))
    quality: Mapped[int | None] = mapped_column(Integer, default=None)
    preserve_metadata: Mapped[bool] = mapped_column(default=True)
    # Forward-compatible bag for sharpening/denoise settings, so adding an
    # option does not require a migration.
    enhance_options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # -- storage (never serialised) ------------------------------------
    input_path: Mapped[str] = mapped_column(Text)
    output_path: Mapped[str | None] = mapped_column(Text, default=None)
    thumbnail_path: Mapped[str | None] = mapped_column(Text, default=None)

    # -- input metadata -------------------------------------------------
    input_width: Mapped[int] = mapped_column(Integer)
    input_height: Mapped[int] = mapped_column(Integer)
    input_bytes: Mapped[int] = mapped_column(Integer)
    input_format: Mapped[str] = mapped_column(String(8))
    original_filename: Mapped[str | None] = mapped_column(Text, default=None)

    # -- output metadata ------------------------------------------------
    output_width: Mapped[int | None] = mapped_column(Integer, default=None)
    output_height: Mapped[int | None] = mapped_column(Integer, default=None)
    output_bytes: Mapped[int | None] = mapped_column(Integer, default=None)

    # -- execution provenance -------------------------------------------
    device: Mapped[str | None] = mapped_column(String(8), default=None)
    tile_size: Mapped[int | None] = mapped_column(Integer, default=None)
    passes: Mapped[int | None] = mapped_column(Integer, default=None)
    processing_ms: Mapped[int | None] = mapped_column(Integer, default=None)

    # -- failure --------------------------------------------------------
    error_code: Mapped[str | None] = mapped_column(String(48), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    error_technical: Mapped[str | None] = mapped_column(Text, default=None)

    # -- cancellation ---------------------------------------------------
    # Cooperative: the tiling loop checks this between tiles (§ 9).
    cancel_requested: Mapped[bool] = mapped_column(default=False)

    # -- timestamps ------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    __table_args__ = (
        # History is listed newest-first, and the sweeper scans by expiry.
        Index("ix_jobs_created_at", "created_at"),
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_expires_at", "expires_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Job {self.id} {self.status} {self.model_name} x{self.scale}>"
