from termcolor import colored


def print_in_red(text):
    """Prints the text in red color and in bold"""
    print(colored(
        text, "red", attrs=["bold"]))


def print_in_blue(text):
    """Prints the text in blue color and in bold"""
    print(colored(
        text, "blue", attrs=["bold"]))
