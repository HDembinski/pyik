import os
import contextlib
import numpy as np
from numpy.testing import assert_allclose, assert_equal


@contextlib.contextmanager
def pushd(new):
    old = os.getcwd()
    os.chdir(str(new))
    yield
    os.chdir(old)


def log(x, mode="r"):
    return open(str(x), mode)
    
    
def test_pmap():
    from pyik.performance import pmap
    assert pmap(lambda x: 2 * x, [1, 2, 3]) == [2, 4, 6]    
    assert pmap(lambda x, y: x * y, (1, 2, 3), [3, 4, 5]) == [3, 8, 15]
    assert_equal(pmap(lambda x: 2 * x, np.ones(3)), (2, 2, 2))


def test_cached_at(tmpdir):
    from pyik.performance import cached_at
    dbfilename = str(tmpdir.join("foo.db"))
    @cached_at(dbfilename)
    def func(x):
        with log(tmpdir.join("log"), "a") as l:
            l.write("x") # leave trace of call
        return 2 * x
    assert func(2) == 4
    assert func(2) == 4
    assert log(tmpdir.join("log")).read() == "x" # only one call


def test_cached(tmpdir):
    from pyik.performance import cached
    @cached
    def func(x):
        with log(tmpdir.join("log"), "a") as l:
            l.write("x") # leave trace of call
        return 2 * x
    with pushd(tmpdir):
        assert func(2) == 4
        assert func(2) == 4
    assert log(tmpdir.join("log")).read() == "x" # only one call
