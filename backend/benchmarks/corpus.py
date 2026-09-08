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
