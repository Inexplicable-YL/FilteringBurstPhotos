import time
from contextlib import contextmanager


@contextmanager
def timer(name="block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{name} took {end - start:.6f} seconds")
