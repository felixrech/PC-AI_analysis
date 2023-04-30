"""Module providing a decorator to memoize function results."""

import sys
from tqdm.auto import tqdm
from flask_caching import Cache
from typing import Callable, Iterable

import dash
from flask import Flask


# Get dash app if it exists
try:
    app = dash.get_app().server
# Otherwise create a flask app
except Exception:
    print("Warning: Fallback to custom flask app for caching")
    app = Flask("fake_app")
# Initialize file cache
cache = Cache(
    app,
    config={
        "CACHE_TYPE": "FileSystemCache",
        "CACHE_DIR": "cache/",
        # Do not limit how many or how long functions are cached
        "CACHE_THRESHOLD": 2**31 - 1,
        "CACHE_DEFAULT_TIMEOUT": 2**31 - 1,
    },
)


def memoize(func: Callable) -> Callable:
    """Memoize function using the initialized cache.

    Parameters
    ----------
    func : Callable
        Arbitrary function.

    Returns
    -------
    Callable
        Memoized function.
    """
    return cache.memoize()(func)  # type: ignore[no-any-return]


def progress_bar(
    iterable: Iterable, desc: str, initial: int = 0, total: int = 0
) -> Iterable:
    """Create a progress report using tqdm for given iterable.

    Parameters
    ----------
    iterable : Iterable
        Arbitrary iterable.
    desc : str
        String to show with progress.
    initial : int, optional
        Initial index, by default 0.
    total : int, optional
        Number of elements in iterable, by default 0, i.e. infer from iterable.

    Returns
    -------
    Iterable
        Tqdm-wrapped iterable.
    """
    total_arg = dict() if total == 0 else dict(total=total)
    return tqdm(  # type: ignore[no-any-return,call-overload]
        list(iterable),
        desc=desc.ljust(15),
        bar_format="{desc}{percentage:3.0f}%",
        initial=initial,
        **total_arg
    )


def delete_last_line() -> None:
    """Delete the last line printed to the console, e.g. an unfinished progress_bar.

    Adapted from code by Aniket Navlur (https://stackoverflow.com/a/52590238).
    """
    # Move cursor to last line
    sys.stdout.write("\x1b[1A")

    # Delete last line
    sys.stdout.write("\x1b[2K")
