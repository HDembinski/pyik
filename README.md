# PyIK - The Python Instrument Kit

[![Build Status](https://travis-ci.org/HDembinski/pyik.svg?branch=master)](https://travis-ci.org/HDembinski/pyik)

This package provides tools to simplify common analysis tasks in particle physics. The main goal is to provide functionality which is currently missing in numpy, scipy, and matplotlib.

The tools are grouped by topic into several modules which are listed below. Most of them depend on external modules which are not shipped with Python. The respective dependencies are also listed.

## Installation

PyIK supports Python2 and Python3. Install it with `pip`.
```
pip install --user pyik
```

## Content

* *corsika*: Tools to work with CORSIKA files
* *ellipse*: Fast and robust fit of an ellipse to noisy data
* *fit*: Classes and functions for function minimization __Requires nlopt__
* *locked_shelve*: Functionality to read shelve files and prevent write collisions
* *misc*: Miscellanious helper functions/classes that cannot be grouped into any of the other submodules
* *mplext*: Missing plotting tools in matplotlib __Requires matplotlib__
* *numpyext*: Missing numerical tools in numpy __Requires numpy, scipy__
* *performance*: Tools to increase performance (e.g. the cached decorator and pmap for easy parallelization)
* *rootext*: Convert Python to ROOT objects and vice-versa __Requires ROOT__
* *time_conversion*: Contains tools to convert between UTC and GPS

## Notes

This packages also contains some a directory with working examples
to copy-paste from.

## Authors

* Ariel Bridgeman
* Hans Dembinski (maintainer)
* Benjamin Fuchs
* Detlef Maurel
* Daniela Mockler
* Alexander Schulz
* Felix Werner
