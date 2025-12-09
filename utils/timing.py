# utils/timing.py

"""
Utility functions for measuring and logging execution time of code blocks and functions.

Usage:
    from utils.timing import time_block, time_function

    # Context manager
    with time_block("data processing"):
        # your code here
        pass

    # Decorator
    @time_function
    def my_function():
        # your code here
        pass
"""

import time
from functools import wraps
from utils.logger import logger


def log_runtime(start: float, label: str) -> None:
    """Log the elapsed time since start in a human-readable format."""
    elapsed = time.perf_counter() - start
    mins = elapsed / 60

    if mins < 60:
        logger.info(f"{label} completed in {mins:.2f} minutes")
    else:
        hours = int(mins // 60)
        rem = mins % 60
        if rem > 0:
            logger.info(f"{label} completed in {hours} hour(s) {rem:.1f} minutes")
        else:
            logger.info(f"{label} completed in {hours} hour(s)")


class time_block:
    """Context manager for timing code blocks.

    Attributes:
        label: Description of the timed block
        elapsed: Time elapsed in seconds (available after block completes)
    """

    def __init__(self, label: str):
        self.label = label
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        log_runtime(self.start, self.label)


def time_function(func):
    """Decorator to time function execution.

    Usage:
        @time_function
        def my_function(arg1, arg2):
            # function code
            pass
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log_runtime(start, func.__name__)
        return result

    return wrapper
