# -*- coding: utf-8 -*-
"""
Contains miscellanious helper functions/classes.

Please try to add functions to the other files first if they can be topically grouped.

"""


def cprint(pstr, cstr="white"):
    """
    Print with color specified in cstr and fall back to normal print if fabric
    not installed.
    """

    try:
        import fabric.api as fab
        from fabric.colors import red, green, blue, yellow, cyan, magenta, white
        cmap = {"white": white, "yellow": yellow, "cyan": cyan, "green": green, "blue": blue,
                "red": red, "magenta": magenta, "g": green, "b": blue, "w": white, "r": red,
                "m": magenta, "c": cyan, "y": yellow}
        fab.puts(cmap[cstr.lower()](pstr))
    except:
        print(pstr)

