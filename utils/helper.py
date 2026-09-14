#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/14 15:46
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   helper.py
# @Desc     :

from random import getstate, setstate
from random import seed as rnd_seed
from time import perf_counter
from typing import Self

WIDTH: int = 64


class Beautifier:
    """ beautifying code blocks using a context manager """

    def __init__(self, desc: str | None = None) -> None:
        """
        Initialise the Beautifier class

        :param desc: the description of a beautifier
        """
        self._desc = desc

    def __enter__(self) -> Self:
        """ Start the beautifier """
        print("*" * WIDTH)
        print(f"The block named {self._desc!r} is starting:")
        print("-" * WIDTH)
        return self

    def __exit__(self, *args) -> None:
        """ Stop the beautifier """
        print("-" * WIDTH)
        print(f"The block named {self._desc!r} has completed.")
        print("*" * WIDTH)
        print()


class Timer:
    """ timing code blocks using a context manager """

    def __init__(self, desc: str | None = None, *, precision: int = 5):
        """
        Initialise the Timer class

        :param desc: the description of a timer
        :param precision: the number of decimal places to round the elapsed time
        """
        if precision < 0:
            raise ValueError("Precision must be >= 0")

        self._desc = desc
        self._precision = precision
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> Self:
        """ Start the timer """
        self._start = perf_counter()
        print("*" * WIDTH)
        print(f"{self._desc!r} has started.")
        print("-" * WIDTH)
        return self

    def __exit__(self, *args) -> None:
        """ Stop the timer and calculate the elapsed time """
        self._end = perf_counter()
        self._elapsed = self._end - self._start

        print("-" * WIDTH)
        print(f"{self._desc!r} took {self._elapsed:.{self._precision}f} seconds.")
        print("*" * WIDTH)

    def __repr__(self) -> str:
        """ Return a string representation of the timer """
        _desc: str = self._desc or "Timer"
        if self._elapsed != 0.0:
            return f"{_desc!r} took {self._elapsed:.{self._precision}f} seconds."
        return f"{_desc!r} has NOT started."


class RandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, desc: str, *, seed: int = 27, tick_tock: bool = False) -> None:
        """
        Initialise the RandomSeed class

        :param desc: the description of a random seed
        :param seed: the seed value to be set
        :param tick_tock: whether to time the random seed setting
        """
        if seed < 1 or seed > 10_000:
            raise ValueError("Seed must be between 1 and 10,000")

        self._desc = desc
        self._curr_seed = seed
        self._prev_seed = None
        self._tick = tick_tock

        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self) -> Self:
        """ Set the random seed """
        if self._tick:
            self._start = perf_counter()

        # Save the previous random seed state
        self._prev_seed = getstate()

        # Set the new random seed
        rnd_seed(self._curr_seed)

        print("*" * WIDTH)
        print(f"{self._desc!r} has been set to {self._curr_seed}.")
        print("-" * WIDTH)
        return self

    def __exit__(self, *args) -> Self:
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._prev_seed is not None:
            setstate(self._prev_seed)

        # Calculate elapsed time if measuring
        if self._tick:
            self._end = perf_counter()
            self._elapsed = self._end - self._start

        print("-" * WIDTH)
        print(f"{self._desc!r} has been restored to previous randomness.")
        if self._tick:
            elapsed_time: str = self._format_time(self._elapsed)
            print(f"{self._desc!r} took {elapsed_time}.")
        print("*" * WIDTH)
        print()
        return False

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format time breakdown from seconds to days, hours, minutes, and seconds

        :param seconds: time in seconds
        :return: formatted time breakdown string
        """
        if seconds < 1.0:
            return f"{seconds * 1000:.1f} ms"

        _days: int = int(seconds // 86_400)
        _hours: int = int((seconds % 86_400) // 3_600)
        _minutes: int = int((seconds % 3_600) // 60)
        _secs: float = seconds % 60

        _parts: list[str] = []
        if _days > 0:
            _parts.append(f"{_days} days")
        if _hours > 0:
            _parts.append(f"{_hours} hours")
        if _minutes > 0:
            _parts.append(f"{_minutes} minutes")
        if _secs > 0 or not _parts:
            _parts.append(f"{_secs:.2f} seconds")
        return " ".join(_parts)

    def __repr__(self) -> str:
        """ Return a string representation of the random seed """
        _base: str = f"PythonRandomSeed({self._desc!r}, seed={self._curr_seed})"
        if self._tick and self._elapsed > 0.0:
            _base += f", Elapsed Time: {self._elapsed:.2f}s"
        return _base
