from __future__ import annotations

from typing import TYPE_CHECKING

import imagehash
from anyio import to_thread

if TYPE_CHECKING:
    from PIL import Image

HASH_SIZE = 8


async def aphash(image: Image.Image) -> str:
    """Compute a perceptual hash for the given image path.

    The function loads the image via :func:`load_image_for_hash`, which handles
    standard formats as well as RAW files (when ``rawpy`` is available).
    """

    phash = await to_thread.run_sync(imagehash.phash, image, HASH_SIZE)
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
