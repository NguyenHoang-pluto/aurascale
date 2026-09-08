"""The benchmark corpus: what to measure, and where it comes from.

No images are committed and none are downloaded. Photographs carry licences,
and a benchmark whose corpus arrives over the network is not reproducible
anyway. The repository holds the *shape* of the corpus; the images are supplied
locally by whoever runs it.

That is also why there is no ground truth here. Nothing in this package knows
what the "right" output looks like, and nothing invents a target to score
against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The categories the investigation called for. Each stresses a different way
# detail can be lost, which is the point of having more than one.
CATEGORIES: tuple[str, ...] = (
    "landscape-detail",
    "portrait-skin",
    "text-signage",
    "foliage-texture",
    "low-light-noise",
)

CATEGORY_NOTES: dict[str, str] = {
    "landscape-detail": "Fine high-frequency structure across the whole frame.",
    "portrait-skin": "Smooth gradients where over-denoising and halos show first.",
    "text-signage": "Hard edges with known ground truth to the eye; ringing is obvious.",
    "foliage-texture": "Dense stochastic texture - the first thing denoising erases.",
    "low-light-noise": "Real sensor noise, to separate denoising from detail loss.",
}

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Default location. Git-ignored by intent; see README.md in this package.
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


@dataclass(frozen=True, slots=True)
class CorpusImage:
    """One input image, and the category it was filed under."""

    category: str
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


def category_dirs(root: Path = DEFAULT_CORPUS_DIR) -> dict[str, Path]:
    """Where each category's images are expected to live."""
    return {category: root / category for category in CATEGORIES}


def load_corpus(root: Path = DEFAULT_CORPUS_DIR) -> list[CorpusImage]:
    """Every supplied image, in a stable order.

    Missing categories are simply absent rather than an error: a run over three
    of the five is still a useful run, and the report says which were present.
    Sorted so two runs enumerate the same images in the same order.
    """
    found: list[CorpusImage] = []

    for category, directory in category_dirs(root).items():
        if not directory.is_dir():
            continue

        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(CorpusImage(category=category, path=path))

    return found


def describe_corpus(root: Path = DEFAULT_CORPUS_DIR) -> str:
    """A human summary of what is present and what is missing."""
    images = load_corpus(root)
    counts = dict.fromkeys(CATEGORIES, 0)
    for image in images:
        counts[image.category] += 1

    lines = [f"corpus root: {root}"]
    for category in CATEGORIES:
        count = counts[category]
        state = f"{count} image(s)" if count else "MISSING - supply images to measure this"
        lines.append(f"  {category:18} {state}")

    return "\n".join(lines)


# --------------------------------------------------------------- the manifest

MANIFEST_NAME = "manifest.json"

#: Fields every manifest entry must carry. Provenance is not optional: an
#: image whose licence or source is unrecorded cannot be used, and guessing
#: either would be worse than dropping it.
REQUIRED_FIELDS: tuple[str, ...] = (
    "filename",
    "category",
    "source",
    "license",
    "dimensions",
)


class ManifestError(ValueError):
    """The manifest is missing, malformed, or describes something absent."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One corpus image, as the manifest describes it."""

    filename: str
    category: str
    source: str
    license: str
    width: int
    height: int
    notes: str = ""
    attribution: str = ""
    license_url: str = ""

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


def manifest_path(root: Path = DEFAULT_CORPUS_DIR) -> Path:
    return root / MANIFEST_NAME


def parse_manifest(payload: object, *, root: Path = DEFAULT_CORPUS_DIR) -> list[ManifestEntry]:
    """Validate a decoded manifest into entries.

    Strict on purpose. A benchmark that silently skips a malformed entry
    reports on a corpus different from the one it claims, and the difference is
    invisible in the results.
    """
    if not isinstance(payload, dict):
        raise ManifestError(f"manifest must be an object, got {type(payload).__name__}")

    images = payload.get("images")
    if not isinstance(images, list):
        raise ManifestError("manifest has no 'images' list")

    entries: list[ManifestEntry] = []
    for index, raw in enumerate(images):
        if not isinstance(raw, dict):
            raise ManifestError(f"image {index} is not an object")

        missing = [field for field in REQUIRED_FIELDS if not raw.get(field)]
        if missing:
            raise ManifestError(f"image {index} is missing {', '.join(missing)}")

        category = str(raw["category"])
        if category not in CATEGORIES:
            raise ManifestError(f"image {index} has unknown category {category!r}")

        dimensions = raw["dimensions"]
        if not isinstance(dimensions, dict) or "width" not in dimensions:
            raise ManifestError(f"image {index} has malformed dimensions")

        entries.append(
            ManifestEntry(
                filename=str(raw["filename"]),
                category=category,
                source=str(raw["source"]),
                license=str(raw["license"]),
                width=int(dimensions["width"]),
                height=int(dimensions["height"]),
                notes=str(raw.get("notes", "")),
                attribution=str(raw.get("attribution", "")),
                license_url=str(raw.get("license_url", "")),
            )
        )

    return entries


def load_manifest(
    root: Path = DEFAULT_CORPUS_DIR, *, require_files: bool = True
) -> list[ManifestEntry]:
    """Read and validate the manifest, and check the files it names exist."""
    path = manifest_path(root)
    if not path.is_file():
        raise ManifestError(
            f"no {MANIFEST_NAME} in {root}. Run `python -m benchmarks.fetch_corpus` first."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc

    entries = parse_manifest(payload, root=root)

    if require_files:
        absent = [entry.filename for entry in entries if not (root / entry.filename).is_file()]
        if absent:
            raise ManifestError(f"manifest names files that are not present: {', '.join(absent)}")

    return entries
