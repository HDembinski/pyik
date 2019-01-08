from pyik.performance import cached_at, cached
import os
import contextlib


@contextlib.contextmanager
def pushd(new):
    old = os.getcwd()
    os.chdir(str(new))
    yield
    os.chdir(old)


def log(x, mode="r"):
    return open(str(x), mode)
    

def test_cached_at(tmpdir):
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
    @cached
    def func(x):
        with log(tmpdir.join("log"), "a") as l:
            l.write("x") # leave trace of call
        return 2 * x
    with pushd(tmpdir):
        assert func(2) == 4
        assert func(2) == 4
    with log(tmpdir.join("log")) as l:
        assert l.read() == "x" # only one call
