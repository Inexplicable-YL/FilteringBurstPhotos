from __future__ import annotations

from pathlib import Path

import imagehash

from .image_raw import load_image_for_hash

HASH_SIZE = 8


def compute_phash(path: Path) -> str:
    """Compute a perceptual hash for the given image path.

    The function loads the image via :func:`load_image_for_hash`, which handles
    standard formats as well as RAW files (when ``rawpy`` is available).
    """

    image = load_image_for_hash(path)
    try:
        phash = imagehash.phash(image, hash_size=HASH_SIZE)
    finally:
        image.close()
    return phash.__str__()


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Return the Hamming distance between two perceptual hash hex strings."""

    image_hash_a = imagehash.hex_to_hash(hash_a)
    image_hash_b = imagehash.hex_to_hash(hash_b)
    return image_hash_a - image_hash_b


def hash_similarity(hash_a: str, hash_b: str) -> float:
    """Compute similarity percentage between two hashes (100 = identical)."""

    distance = hamming_distance(hash_a, hash_b)
    max_distance = HASH_SIZE * HASH_SIZE
    return (1 - distance / max_distance) * 100
