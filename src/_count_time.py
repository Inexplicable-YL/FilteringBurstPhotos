import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timer(name="block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        text = f"Elapsed time for {name}: {end - start:.4f} seconds"
        logger.info(text)
