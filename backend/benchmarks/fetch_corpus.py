"""Fetch the real-photograph corpus from Wikimedia Commons.

Run once to populate `benchmarks/corpus/`. The images are not committed - they
carry licences and attribution obligations, and a corpus fetched at build time
is reproducible from the manifest instead.

Two decisions worth stating, because both affect what the benchmark can
conclude:

  * **Originals, cropped, never scaled.** Downscaling a photograph averages
    sensor noise away, and sensor noise is precisely what a denoise benchmark
    measures. A crop keeps every pixel exactly as the camera recorded it; a
    resize would quietly answer the question before the experiment ran.
  * **Licence and authorship come from the API**, not from this file. Whatever
    Commons reports is what the manifest records, so provenance cannot drift
    from the source.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AuraScale-benchmark/0.1 (image-quality research)"

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
MANIFEST = CORPUS_DIR / "manifest.json"

# Roughly how many pixels each corpus image should end up at. Large enough to
# hold real texture and noise, small enough that five denoise arms across the
# whole corpus finish in minutes rather than an afternoon.
TARGET_PIXELS = 2_000_000

# Licences this corpus will accept. Anything else is skipped rather than
# quietly included.
ALLOWED_LICENCES = ("cc0", "public domain", "cc by", "cc by-sa")


@dataclass(frozen=True, slots=True)
class Wanted:
    """One curated Commons file and the category it stands in for."""

    category: str
    title: str
    notes: str


WANTED: tuple[Wanted, ...] = (
    Wanted(
        "landscape-detail",
        "File:Mountains in snow, Mountain lake, Chola Valley, Nepal, Himalayas.jpg",
        "Distant rock and snow texture; fine high-frequency structure across the frame.",
    ),
    Wanted(
        "portrait-skin",
        "File:Headshot Prof Shafi AHmed 001 photo.jpg",
        "Skin gradients and hair. Over-denoising shows here first as waxy skin.",
    ),
    Wanted(
        "text-signage",
        "File:SE Ankeny Street sign, Portland, Oregon.jpg",
        "Hard lettering edges; ringing and halos are obvious to the eye.",
    ),
    Wanted(
        "foliage-texture",
        "File:(Cephalomanes fern foliage close-up in Espiritu Santo, Vanuatu) - DPLA - "
        "970b3b811cfb7b136fe34d425d0e9741.jpg",
        "Dense stochastic leaf texture - the first thing denoising erases.",
    ),
    Wanted(
        "low-light-noise",
        "File:Széchenyi Chain Bridge in Budapest at night.jpg",
        "Night exposure with real shadow noise, to separate denoising from detail loss.",
    ),
)


# Commons rate-limits anonymous clients. A pause between requests is politer
# than retrying into a 429, and this script runs once.
REQUEST_PAUSE_SECONDS = 3.0
RETRY_ATTEMPTS = 4


def _open(url: str, timeout: int = 120) -> bytes:
    """Fetch a URL, backing off when Commons asks us to slow down."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload: bytes = response.read()
            return payload
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == RETRY_ATTEMPTS - 1:
                raise
            wait = REQUEST_PAUSE_SECONDS * (2**attempt) * 3
            print(f"    rate limited; waiting {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _get(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode({**params, "format": "json"})
    payload: dict[str, Any] = json.loads(_open(f"{API}?{query}", timeout=60))
    return payload


def strip_html(value: str) -> str:
    """Commons returns attribution as HTML; the manifest wants text."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value)).strip()


def describe(title: str) -> dict[str, Any]:
    """Licence, author and original URL, straight from the API."""
    payload = _get(
        {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "titles": title,
        }
    )
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    info = (page.get("imageinfo") or [{}])[0]
    meta = info.get("extmetadata", {})

    def field(name: str) -> str:
        return strip_html(str(meta.get(name, {}).get("value", "")))

    return {
        "title": page.get("title", title),
        "url": info.get("url", "").split("?")[0],
        "descriptionurl": info.get("descriptionurl", ""),
        "width": int(info.get("width", 0)),
        "height": int(info.get("height", 0)),
        "license": field("LicenseShortName"),
        "license_url": field("LicenseUrl"),
        "artist": field("Artist"),
        "credit": field("Credit"),
    }


def centre_crop_box(
    width: int, height: int, target: int = TARGET_PIXELS
) -> tuple[int, int, int, int]:
    """A centred box of about `target` pixels, keeping the aspect ratio.

    Returns the whole image when it is already small enough - an image is never
    enlarged, and never resampled.
    """
    if width * height <= target:
        return (0, 0, width, height)

    ratio = (target / (width * height)) ** 0.5
    crop_w = max(1, int(width * ratio))
    crop_h = max(1, int(height * ratio))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return (left, top, left + crop_w, top + crop_h)


def acceptable(licence: str) -> bool:
    lowered = licence.lower()
    return any(token in lowered for token in ALLOWED_LICENCES)


def fetch(wanted: Wanted) -> dict[str, Any] | None:
    from PIL import Image

    meta = describe(wanted.title)
    if not acceptable(meta["license"]):
        print(f"  SKIP {wanted.category}: licence {meta['license']!r} not in the allowed set")
        return None

    directory = CORPUS_DIR / wanted.category
    directory.mkdir(parents=True, exist_ok=True)

    suffix = Path(urllib.parse.unquote(meta["url"])).suffix.lower() or ".jpg"
    destination = directory / f"{wanted.category}{suffix}"

    destination.write_bytes(_open(meta["url"], timeout=300))

    with Image.open(destination) as opened:
        original = opened.size
        box = centre_crop_box(*original)
        cropped = opened.convert("RGB").crop(box)
        # Re-encoded at quality 97 so the crop is not a second lossy generation
        # in any meaningful sense. No resampling has happened.
        cropped.save(destination, format="JPEG", quality=97, subsampling=0)
        final = cropped.size

    print(f"  {wanted.category:18} {original[0]}x{original[1]} -> crop {final[0]}x{final[1]}")

    return {
        "filename": f"{wanted.category}/{destination.name}",
        "category": wanted.category,
        "source": meta["descriptionurl"] or meta["url"],
        "source_file": meta["title"],
        "license": meta["license"],
        "license_url": meta["license_url"],
        "attribution": meta["artist"] or meta["credit"],
        "original_dimensions": {"width": original[0], "height": original[1]},
        "dimensions": {"width": final[0], "height": final[1]},
        "crop_box": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
        "processing": "centre crop only; never resampled, so sensor noise is untouched",
        "notes": wanted.notes,
    }


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"fetching {len(WANTED)} images into {CORPUS_DIR}")

    entries: list[dict[str, Any]] = []
    for index, wanted in enumerate(WANTED):
        if index:
            time.sleep(REQUEST_PAUSE_SECONDS)
        entry = fetch(wanted)
        if entry is not None:
            entries.append(entry)

    MANIFEST.write_text(
        json.dumps(
            {
                "description": (
                    "Real photographs for the AuraScale image-quality benchmark. "
                    "Fetched from Wikimedia Commons; licence and attribution are "
                    "recorded as the API reported them. Images are centre-cropped "
                    "from the originals and never resampled, because downscaling "
                    "would average away the sensor noise the benchmark measures."
                ),
                "generated_by": "benchmarks/fetch_corpus.py",
                "images": entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST} with {len(entries)} entries")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
