#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/14 15:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   decorator.py
# @Desc     :

from collections.abc import Callable
from functools import wraps
from time import perf_counter

WIDTH: int = 64


def beautifier(func):
    """
    Decorator to beautify function execution with start and completion logs.

    :param func: The function to be decorated.
    :return: A wrapped function that adds logging around the original call.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper that executes the function with beautified logs.

        :param args: Positional arguments passed to the original function.
        :param kwargs: Keyword arguments passed to the original function.
        :return: The return value of the original function.
        """
        print("*" * WIDTH)
        print(f"The function named {func.__name__!r} is starting:")
        print("-" * WIDTH)
        result = func(*args, **kwargs)
        print("-" * WIDTH)
        print(f"The function named {func.__name__!r} has completed.")
        print("*" * WIDTH)
        print()
        return result

    return wrapper


def countdown(func):
    """
    Decorator to time function execution and log performance.

    This decorator measures the elapsed time of the decorated function
    and prints a formatted log with the execution duration.

    :param func: The function to be decorated.
    :return: A wrapped function that adds timing and logging around the original call.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Wrapper that executes the function with timing and logs.

        :param args: Positional arguments passed to the original function.
        :param kwargs: Keyword arguments passed to the original function.
        :return: The return value of the original function.
        """
        print("*" * WIDTH)
        print(f"The function named {func.__name__!r} is starting:")
        print("-" * WIDTH)

        time_start = perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            time_end = perf_counter()
            time_elapsed = time_end - time_start
            print("-" * WIDTH)
            print(f"The function named {func.__name__!r} took {time_elapsed:.4f} seconds to complete.")
            print("*" * WIDTH)
            print()

    return wrapper


def clock(desc: str | None = None):
    """
    Decorator to time function execution and log performance.

    This decorator measures the elapsed time of the decorated function
    and prints a formatted log with the execution duration.

    :param desc: Optional custom name for the function shown in logs.
    :return: A wrapped function that adds timing and logging around the original call.
    """

    def decorator(func):
        """
        Inner decorator that wraps the target function with timing logic.

        :param func: The function to be decorated.
        :return: The wrapped function with timing and beautification.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper that executes the function with timing and optional logs.

            :param args: Positional arguments passed to the original function.
            :param kwargs: Keyword arguments passed to the original function.
            :return: The return value of the original function.
            """
            _desc = desc or func.__name__
            _authorise: bool = kwargs.pop("display", True)

            if not _authorise:
                return func(*args, **kwargs)

            print("*" * WIDTH)
            print(f"The function named {_desc!r} is starting:")
            print("-" * WIDTH)

            time_start = perf_counter()

            try:
                return func(*args, **kwargs)
            finally:
                time_end = perf_counter()
                time_elapsed = time_end - time_start
                print("-" * WIDTH)
                print(f"The function named {_desc!r} took {time_elapsed:.4f} seconds to complete.")
                print("*" * WIDTH)
                print()

        return wrapper

    return decorator


def timer(arg: Callable | str | None = None) -> Callable:
    """
    Decorator to time function execution and log performance.

    This decorator measures the elapsed time of the decorated function
    and prints a formatted log with the execution duration.

    :param arg: The function to be decorated or the custom name for the function shown in logs.
    :return: A wrapped function that adds timing and logging around the original call.
    """
    # @timer
    if callable(arg):
        return _build_wrapper(arg, desc=None)

    # @timer() / @timer("desc")
    desc = arg

    def decorator(func):
        return _build_wrapper(func, desc=desc)

    return decorator


def _build_wrapper(func, desc: str | None) -> Callable:
    """
    Inner decorator that wraps the target function with timing logic.
    :param func: The function to be decorated.
    :param desc: The custom name for the function shown in logs.
    :return: The wrapped function with timing and logging.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        _desc = desc or func.__name__
        print("*" * WIDTH)
        print(f"The function named {_desc!r} is starting:")
        print("-" * WIDTH)

        time_start = perf_counter()

        try:
            return func(*args, **kwargs)
        finally:
            time_elapsed = perf_counter() - time_start
            print("-" * WIDTH)
            print(f"The function named {_desc!r} took {time_elapsed:.4f} seconds to complete.")
            print("*" * WIDTH)
            print()

    return wrapper
