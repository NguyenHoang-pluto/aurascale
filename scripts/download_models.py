#!/usr/bin/env python
"""Download Real-ESRGAN weights into the models directory.

    backend/.venv/Scripts/python scripts/download_models.py --all
    backend/.venv/Scripts/python scripts/download_models.py RealESRGAN_x4plus

Weights are release artifacts of the Real-ESRGAN project, not something this
repository redistributes. They are fetched over HTTPS from the URLs recorded in
`models/manifest.json` and verified against the SHA-256 digest recorded there.

The digests start out null. `--trust-first-download` accepts whatever the
official URL serves and writes the digest back into the manifest, which is how
a digest gets recorded in the first place; every later download is checked
against it. Review the diff before committing a recorded digest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.core.exceptions import PixelForgeError  # noqa: E402
from app.services.model_download_service import ModelDownloadService  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402

MB = 1024 * 1024


def _progress(received: int, total: int | None) -> None:
    if total:
        percent = received / total * 100
        print(f"\r    {received / MB:6.1f} / {total / MB:.1f} MB  ({percent:5.1f}%)", end="")
    else:  # pragma: no cover - servers that omit Content-Length
        print(f"\r    {received / MB:6.1f} MB", end="")


def record_digest(manifest_path: Path, model_id: str, digest: str) -> None:
    """Write a verified digest back into the manifest.

    Rewritten with the same indentation the file already uses so the diff shows
    only the digest that changed.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    for entry in payload["models"]:
        if entry["id"] == model_id:
            entry["sha256"] = digest
            break

    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="model ids; omit with --all")
    parser.add_argument("--all", action="store_true", help="download every model")
    parser.add_argument(
        "--trust-first-download",
        action="store_true",
        help="accept a download with no recorded digest, and record it",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    registry = ModelService(settings)
    downloader = ModelDownloadService(settings, registry)

    if arguments.all:
        wanted = [status.entry.id for status in registry.list()]
    elif arguments.models:
        # A denoise-capable model is useless without its pair, so pull both.
        wanted = []
        for model_id in arguments.models:
            for status in downloader.required_models(model_id):
                if status.entry.id not in wanted:
                    wanted.append(status.entry.id)
    else:
        parser.error("name at least one model, or pass --all")

    failures = 0

    for model_id in wanted:
        status = registry.get(model_id)
        if status.downloaded:
            size = (status.size_bytes or 0) / MB
            print(f"  {model_id:32} already present ({size:.1f} MB)")
            continue

        print(f"  {model_id:32} downloading…")
        try:
            result = downloader.download(
                status,
                trust_first_download=arguments.trust_first_download,
                on_progress=_progress,
            )
        except PixelForgeError as error:
            print(f"\r  {model_id:32} FAILED: {error.message}")
            if error.technical:
                print(f"    {error.technical}")
            failures += 1
            continue

        print(f"\r  {model_id:32} {result.size_bytes / MB:6.1f} MB  sha256={result.sha256}")

        if status.entry.sha256 is None:
            record_digest(registry.manifest_path, model_id, result.sha256)
            print(f"  {'':32} digest recorded in manifest.json")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
