import functools
import time
from termcolor import colored


def timer(function):
    """Print the runtime of the decorated fucntion."""
    @functools.wraps(function)
    def wrapper_timer(*args, **kwargs):

        start = time.perf_counter()
        print(
            f"Starting execution of {function.__name__!r} function, Please wait...")
        value = function(*args, **kwargs)
        end = time.perf_counter()
        runtime = end - start
        print(
            colored(f"Finished execution of {function.__name__!r} function in {runtime:.3f} secs\n", "blue", attrs=["bold"]))
        return value

    return wrapper_timer


def conditional_timer(function):
    """Print the runtime of the decorated fucntion only when condition is true."""
    @functools.wraps(function)
    def wrapper_timer(*args, condition: bool):
        try:
            start = time.perf_counter()
            if condition:
                print(
                    f"Starting execution of {function.__name__!r} function, Please wait...")
            value = function(*args, condition)
            end = time.perf_counter()
            runtime = end - start
            if condition:
                print(
                    colored(f"Finished execution of {function.__name__!r} function in {runtime:.3f} secs\n", "blue", attrs=["bold"]))
                return value
        except Exception as e:
            print(
                colored(f"Error in execution of {function.__name__!r}:\n" + str(e), "red", attrs=["bold"]))
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
                colored(f"Error in execution of {function.__name__!r}:\n" + str(e), "red", attrs=["bold"]))
    return wrapper_timer
