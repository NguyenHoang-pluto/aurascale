"""Model registry.

Reads `models/manifest.json` — the single source of truth for which weights
exist, where they come from and how their architecture is parameterised — and
reports which are present on disk.

This service knows nothing about PyTorch. Loading weights is the ModelManager's
job in Phase 6; keeping the registry separate means the model list is available
even when torch is broken or no weights are downloaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.core.exceptions import ModelNotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelEntry(BaseModel):
    """One entry from the manifest, validated on load."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    name: str
    description: str
    arch: str
    scale: int
    file: str
    url: str
    sha256: str | None = None
    arch_params: dict[str, Any] = Field(default_factory=dict)
    supports_denoise: bool = False
    # The denoising counterpart blended by DNI weight interpolation.
    denoise_pair: str | None = None
    # Weight sets that are not offered on their own, such as the wdn variant.
    selectable: bool = True


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """A manifest entry plus its on-disk state."""

    entry: ModelEntry
    downloaded: bool
    size_bytes: int | None
    path: Path


class ModelService:
    """Reads the manifest and reports availability."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def manifest_path(self) -> Path:
        return self._settings.models_dir / "manifest.json"

    def _load_entries(self) -> list[ModelEntry]:
        return _read_manifest(self.manifest_path)

    def list(self, *, selectable_only: bool = False) -> list[ModelStatus]:
        """All registered models, optionally hiding non-selectable weight sets."""
        statuses: list[ModelStatus] = []

        for entry in self._load_entries():
            if selectable_only and not entry.selectable:
                continue

            path = self._settings.models_dir / entry.file
            exists = path.is_file()
            statuses.append(
                ModelStatus(
                    entry=entry,
                    downloaded=exists,
                    size_bytes=path.stat().st_size if exists else None,
                    path=path,
                )
            )

        return statuses

    def get(self, model_id: str) -> ModelStatus:
        """One model by id.

        Raises ModelNotFoundError rather than returning None: an unknown model
        id is a client error with a specific message, not an absent value.
        """
        for status in self.list():
            if status.entry.id == model_id:
                return status

        known = ", ".join(sorted(s.entry.id for s in self.list() if s.entry.selectable))
        raise ModelNotFoundError(
            f"There is no model called {model_id!r}.",
            technical=f"known models: {known}",
            context={"requested": model_id},
        )

    def default_model_id(self) -> str:
        """Configured default, validated against the manifest at call time."""
        self.get(self._settings.default_model)
        return self._settings.default_model


@lru_cache(maxsize=4)
def _read_manifest(path: Path) -> list[ModelEntry]:
    """Parse and validate the manifest.

    Cached by path: the manifest is a static asset, and re-reading it on every
    request would be wasteful. Tests use distinct temporary paths, so the cache
    never leaks between them.
    """
    if not path.is_file():
        raise ModelNotFoundError(
            "The model manifest is missing, so no models are available.",
            technical=f"expected at {path.name}",
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = payload["models"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelNotFoundError(
            "The model manifest could not be read.",
            technical=f"{type(exc).__name__}: {exc}",
        ) from exc

    entries = [ModelEntry.model_validate(item) for item in raw_entries]
    logger.debug("manifest loaded", extra={"models": len(entries)})
    return entries
