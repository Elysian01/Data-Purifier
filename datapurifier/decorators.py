import functools
import time
from termcolor import colored


def timer_and_exception_handler(function):
    """Print the runtime of the decorated fucntion."""
    @functools.wraps(function)
    def wrapper_timer(*args, **kwargs):
        try:
            start = time.perf_counter()
            print(
                f"Starting execution of {function.__name__!r} function, Please wait...")
            value = function(*args, **kwargs)
            end = time.perf_counter()
            runtime = end - start
            print(
                colored(f"Finished execution of {function.__name__!r} function in {runtime:.3f} secs\n", "blue", attrs=["bold"]))
            return value
        except Exception as e:
            print(
                colored(f"Error in execution of {function.__name__!r}:\n" + e, "red", attrs=["bold"]))
    return wrapper_timer


def exception_handler(function):
    """Handles Error of decorated fucntion."""
    @functools.wraps(function)
    def wrapper_timer(*args, **kwargs):
        try:
            value = function(*args, **kwargs)
            return value
        except Exception as e:
            print(
                colored(f"Error in execution of {function.__name__!r}:\n" + e, "red", attrs=["bold"]))
    return wrapper_timer
