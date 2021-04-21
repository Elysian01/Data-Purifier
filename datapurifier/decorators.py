import functools
import time
from termcolor import colored


def timer(function):
    """Print the runtime of the decorated fucntion."""
    @functools.wraps(function)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        value = function(*args, **kwargs)
        end = time.perf_counter()
        runtime = end - start
        print(
            colored(f"Finished execution of {function.__name__!r} function in {runtime:.3f} secs", "blue"))
        return value
    return wrapper_timer
