#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/14 15:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :

"""
****************************************************************
Utility Module - Comprehensive Toolkit
----------------------------------------------------------------
This module provides a comprehensive suite of utility functions
and classes designed for general data processing tasks.
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .decorator import beautifier, clock, countdown, timer
from .helper import Beautifier, RandomSeed, Timer
from .highlighter import (
    black,
    blue,
    bold,
    cyan,
    green,
    invert,
    lines,
    purple,
    red,
    sharps,
    stars,
    strikethrough,
    underline,
    white,
    yellow,
)

__all__ = [
    "beautifier",
    "countdown", "clock", "timer",

    "Beautifier",
    "Timer",
    "RandomSeed",

    "black", "red", "green", "yellow", "blue", "purple", "cyan", "white",
    "bold", "underline", "invert", "strikethrough",
    "stars", "lines", "sharps",
]
