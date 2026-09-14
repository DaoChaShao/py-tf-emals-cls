#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2026/9/14 15:45
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   highlighter.py
# @Desc     :

from typing import Any

WIDTH: int = 64


def black(text: Any) -> str:
    """
    Highlight text with Black

    :param text: text to be highlighted
    :return: text is highlighted with Black
    """
    if isinstance(text, int):
        return f"\033[1;30m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;30m{text:9.5f}\033[0m"
    else:
        return f"\033[1;30m{text}\033[0m"


def red(text: Any) -> str:
    """
    Highlight text with Red

    :param text: text to be highlighted
    :return: text is highlighted with Red
    """
    if isinstance(text, int):
        return f"\033[1;31m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;31m{text:9.5f}\033[0m"
    else:
        return f"\033[1;31m{text}\033[0m"


def green(text: Any) -> str:
    """
    Highlight text with Green

    :param text: text to be highlighted
    :return: text is highlighted with Green
    """
    if isinstance(text, int):
        return f"\033[1;32m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;32m{text:9.5f}\033[0m"
    else:
        return f"\033[1;32m{text}\033[0m"


def yellow(text: Any) -> str:
    """
    Highlight text with Yellow

    :param text: text to be highlighted
    :return: text is highlighted with Yellow
    """
    if isinstance(text, int):
        return f"\033[1;33m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;33m{text:9.5f}\033[0m"
    else:
        return f"\033[1;33m{text}\033[0m"


def blue(text: Any) -> str:
    """
    Highlight text with Blue

    :param text: text to be highlighted
    :return: text is highlighted with Blue
    """
    if isinstance(text, int):
        return f"\033[1;34m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;34m{text:9.5f}\033[0m"
    else:
        return f"\033[1;34m{text}\033[0m"


def purple(text: Any) -> str:
    """
    Highlight text with Purple

    :param text: text to be highlighted
    :return: text is highlighted with Purple
    """
    if isinstance(text, int):
        return f"\033[1;35m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;35m{text:9.5f}\033[0m"
    else:
        return f"\033[1;35m{text}\033[0m"


def cyan(text: Any) -> str:
    """
    Highlight text with Cyan

    :param text: text to be highlighted
    :return: text is highlighted with Cyan
    """
    if isinstance(text, int):
        return f"\033[1;36m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;36m{text:9.5f}\033[0m"
    else:
        return f"\033[1;36m{text}\033[0m"


def white(text: Any) -> str:
    """
    Highlight text with White

    :param text: text to be highlighted
    :return: text is highlighted with White
    """
    if isinstance(text, int):
        return f"\033[1;37m{text:12d}\033[0m"
    elif isinstance(text, float):
        return f"\033[1;37m{text:9.5f}\033[0m"
    else:
        return f"\033[1;37m{text}\033[0m"


def bold(text: Any) -> str:
    """
    Bold text

    :param text: text to be bolded
    :return: text is bolded
    """
    return f"\033[1m{text}\033[0m"


def underline(text: Any) -> str:
    """
    Underline text

    :param text: text to be underlined
    :return: text is underlined
    """
    return f"\033[4m{text}\033[0m"


def invert(text: Any) -> str:
    """
    Invert text colour

    :param text: text to be inverted
    :return: text color is inverted
    """
    return f"\033[7m{text}\033[0m"


def strikethrough(text: Any) -> str:
    """
    Strikethrough text

    :param text: text to be strikethrough
    :return: text is strikethrough
    """
    return f"\033[9m{text}\033[0m"


def stars(text: str = "", *, length: int = WIDTH) -> None:
    """
    Draw stars

    :param text: text to be written down between stars
    :param length: how many stars to draw
    :return:
    """
    if text:
        left = (length - len(text)) // 2
        right = length - len(text) - left
        print("*" * left + text + "*" * right)
    else:
        print("*" * length)


def lines(text: str = "", *, length: int = WIDTH) -> None:
    """
    Draw lines

    :param text: text to be written down between short lines
    :param length: how many short lines to draw
    :return:
    """
    if text:
        left = (length - len(text)) // 2
        right = length - len(text) - left
        print("-" * left + text + "-" * right)
    else:
        print("-" * length)


def sharps(text: str = "", *, length: int = WIDTH) -> None:
    """
    Draw sharps

    :param text: text to be written down between sharps
    :param length: how many sharps to draw
    :return:
    """
    if text:
        left = (length - len(text)) // 2
        right = length - len(text) - left
        print("#" * left + text + "#" * right)
    else:
        print("#" * length)
