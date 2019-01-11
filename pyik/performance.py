# -*- coding: utf-8 -*-
import numpy as np
import warnings
import multiprocessing as mp


def pmap(function, *arguments, **kwargs):
    """
    Parallelized version of map. It calls function on arguments using numprocesses threads.

    Also try the numexpr package or numba, which are easy to use and may give you better
    performance.

    Parameters
    ----------
    function : function
      The function to call.
    arguments : sequence(s)
      One or more sequences, which are passed to the function. If N sequences are
      provided, function is called with N arguments.
    numprocesses : int, optional (default = None)
      The number of processes to keep busy. If None, the number of CPUs is used.
    nchunks: int, optional (default = None)
      The number of chunks in which the arguments are split.
      By default this is set to the number of processes, but sometimes you want
      fewer chunks, so that they do not become too short.

    Returns
    -------
    results : list or ndarray
      The values returned by the function calls, as would be done by
      map(function, *arguments). If async is True, the order of the results is
      arbitrary. If the first of arguments is an ndarray, the result is converted to an
      ndarray.

    Notes
    -----
    Uses multiprocessing queues to create parallel version of map.
    Note that, for pmap to be useful, the function itself (applied on one input) should be time consuming and hard to optimize.
    Pmap is not beneficial in case of a cheap function and huge number of inputs.
    In this case it is much better to run the function in a lower level, e.g. via numpy vectorization.

    Limitations
    -----------
    The function needs to be a pure function, which means that
    it may not manipulate a shared global state during its calls.
    For example, you cannot use a function that computes a sum over the arguments.

    Examples
    --------
    >>> def f(x): return x*x
    >>> pmap(f, (1, 2, 3))
    [1, 4, 9]

    See Also
    --------
    multiprocessing
    """

    assert len(arguments) > 0, "at least one iterable argument is required"
    assert all(np.iterable(args) for args in arguments), "arguments must be iterable"
    length = len(arguments[0])
    assert all(length == len(args) for args in arguments[1:]), "all arguments must have same length"

    numprocesses = kwargs.get("numprocesses", mp.cpu_count())
    nchunks = kwargs.get("nchunks", numprocesses)

    chunksize = kwargs.get("chunksize", None)
    if chunksize is not None:
        warnings.warn("chunksize keyword is deprecated, please use nchunks keyword",
                      DeprecationWarning)
        nchunks = chunksize

    if not kwargs.get("async", True):
        warnings.warn("async == False is deprecated, pmap results are always in "
                      "correct order now",
                      DeprecationWarning)

    nchunks = min(numprocesses, length, nchunks)
    argchunks = [np.array_split(args, nchunks) for args in arguments]

    def worker(f, conn):
        args = conn.recv()
        result = list(map(f, *args))
        conn.send(result)

    pipes = [mp.Pipe() for _ in range(nchunks)]
    procs = [mp.Process(target=worker, args=(function, p[1])) for p in pipes]

    for p in procs:
        p.daemon = True
        p.start()

    for i, p in enumerate(pipes):
        args = [args[i] for args in argchunks]
        p[0].send(args)

    results = [p[0].recv() for p in pipes]

    for p in procs:
        p.join()

    first_arg = arguments[0]
    if isinstance(first_arg, np.ndarray):
        return np.concatenate([x for x in results], axis=0)
    res = []
    for r in results:
        res += r
    return res


def cached(keepOpen=False, lockCacheFile=False, trackCode=True):
    """
    Decorator which caches the result of a slow function in an automatically
    generated file in the current working directory.


    Parameters
    ----------
    keepOpen : bool, optional (default = False)
      Determines whether to keep the cache file open during the whole session.
    lockCacheFile : bool, optional (default = False)
      Locks the cache file during access (necessary when used with pmap)
      don't use this option together with keepOpen, otherwise you will get a dead lock
    trackCode: bool, optional (default = True)
      If true, also changes in the code will be tracked.

    Returns
    -------
    f : decorated function

    Notes
    -----
    This function decorators enhances a function by wrapping code around it
    that implements the caching of the result. See the examples how to use it.
    The decorator is able to cache the results of several individual functions
    in a single script, each called with several individual arguments.

    The cache is useful, for example, if you are a slow routine that generates
    some data which you want to plot. Typically, the plot part needs several
    development cycles to look good. While you develop the display, the slow
    parts of your analysis are quickly supplied by the cache for you.

    The cache file is always created in the current working directory. The
    filename is cache-<script name>-<function name>.pkl. If the decorator
    is used within an interactive session, the filename will be python.cache.
    If you want to place and name the file yourself, use the "cached_at" decorator.

    The decorator detects edits in the cached function via a hash of its byte
    code. If you want to make sure that the cache is updated, please delete
    the cache file manually.

    Limitations
    -----------
    The function is only are allowed to accept and return objects
    that can be handled by the pickle module. Fortunately, that is practically
    everything.

    Examples
    --------
    In order to make a cached version of a slow function, do

    >>> @cached
    ... def slow_function1(a):
    ...     return a
    >>> @cached
    ... def slow_function2(b):
    ...     return b*b
    >>> slow_function1(2)
    2
    >>> slow_function2(2)
    4
    >>> slow_function1(3)
    3

    Any calls to slow_function1 and slow_function2 are transparently cached
    from now on.

    See also
    --------
    shelve
    pickle
    """

    import os
    from types import FunctionType
    from functools import wraps
    import inspect

    # To preserve backwards compatibility, calling without any arguments is
    # still supported. In this case, keepOpen is the function to be cached.
    if isinstance(keepOpen, FunctionType):
        function = keepOpen
        # check for name of script and assume interactive session otherwise
        prog = inspect.getfile(function)
        prog = os.path.splitext(os.path.basename(prog))[
            0] if os.path.exists(prog) else "python"
        cacheFileName = "cache-" + prog + "-" + function.__name__ + ".pkl"
        return cached_at(cacheFileName)(function)

    def decorator(function):
        # check for name of script and assume interactive session otherwise
        prog = inspect.getfile(function)
        prog = os.path.splitext(os.path.basename(prog))[
            0] if os.path.exists(prog) else "python"
        cacheFileName = "cache-" + prog + "-" + function.__name__ + ".pkl"

        @wraps(function)
        @cached_at(cacheFileName, keepOpen, lockCacheFile, trackCode)
        def decorated_function(*args, **kwargs):

            return function(*args, **kwargs)

        return decorated_function

    return decorator


def cached_at(cacheFileName, keepOpen=False, lockCacheFile=False, trackCode=True):
    """
    Decorator which caches the result of a slow function in a file.

    Parameters
    ----------
    cacheFileName : string
      Path and filename of the cache file.
    keepOpen : bool, optional (default = False)
      Determines whether to keep the cache file open during the whole session.
    lockCacheFile : bool, optional (default = False)
      locks the cache file during access (necessary when used with pmap)
      don't use this option together with keepOpen, otherwise you will get a dead lock
    trackCode: bool, optional (default = True)
      If true, also changes in the code will be tracked.

    Returns
    -------
    f : decorated function

    Notes
    -----
    See the decorator "cached" in this module.

    Limitations
    -----------
    See the decorator "cached" in this module.

    Examples
    --------
    In order to make a cached version of a slow function, do

    >>> @cached_at("mycache1.tmp")
    ... def slow_function1(a):
    ...   return a
    >>> @cached_at("mycache2.tmp")
    ... def slow_function2(b):
    ...   return b*b
    >>> slow_function1(2)
    2
    >>> slow_function2(2)
    4
    >>> slow_function1(3)
    3

    See the decorator "cached" in this module for more information.

    See also
    --------
    shelve
    pickle
    """

    from functools import wraps

    if keepOpen and lockCacheFile:
        raise ValueError("keepOpen cannot be used together with lockCacheFile")

    if lockCacheFile:
        from . import locked_shelve as shelve
    else:
        import shelve

    if keepOpen:
        # Open the shelve on function decoration
        _d = shelve.open(cacheFileName, protocol=-1, writeback=False)

    def decorator(function):

        @wraps(function)
        def decorated_function(*args, **kwargs):

            from six.moves import cPickle as pickle
            import six
            import inspect
            import hashlib

            def encode(x):
                if six.PY2:
                    return x
                else:
                    return x.encode("utf-8")

            # Pickle the function arguments to use them as key
            # it is preferable not to include the function name in the pickle
            # when the function name changes, the cache file name changes anyway
            # if the user decides to recall the function and manually recall the
            # cache, it will still work
            key = str(pickle.dumps((args, kwargs), protocol=-1))
            code_hash = hashlib.md5(encode(inspect.getsource(function))).digest()

            if keepOpen:
                d = _d  # Use open shelve
            else:
                d = shelve.open(cacheFileName, protocol=-1, writeback=False)

            create = True  # this variable is necessary, cached output could be anything, also None
            if key in d:
                assert "cache" in d[
                    key], "Your cache might be outdated. Try to delete and create it again!"
                if trackCode:
                    if code_hash in d[key]["code_hash"]:
                        output = d[key]["cache"]
                        create = False
                else:
                    output = d[key]["cache"]
                    create = False

            if create:
                if not keepOpen:
                    d.close()
                output = function(*args, **kwargs)
                if not keepOpen:
                    d = shelve.open(cacheFileName, protocol=-
                                    1, writeback=False)

                if key in d:
                    if pickle.dumps([d[key]["cache"]], protocol=-1) == pickle.dumps([output], protocol=-1):
                        dk = d[key]
                        dk["code_hash"] += [code_hash]
                        d[key] = dk
                    else:
                        d[key] = {"code_hash": [code_hash], "cache": output}
                else:
                    d[key] = {"code_hash": [code_hash], "cache": output}

            if not keepOpen:
                d.close()

            return output

        return decorated_function

    return decorator


def memoized(function):
    """
    Caches the output of a function in memory to increase performance.

    Returns
    -------
    f : decorated function

    Notes
    -----
    This decorator speeds up slow calculations that you need over and over
    in a script. If you want to keep the results of a slow function for
    several script executions, use the "cached" decorator instead
    (which also allows mutable arguments).

    Limitations
    -----------
    Use this decorator only for functions with immutable arguments, like
    numbers, tuples, and strings. The decorator is intended for simple
    mathematical functions and optimized for performance.
    """

    from functools import wraps

    cache = {}

    @wraps(function)
    def decorated_function(*args):

        if args in cache:
            output = cache[args]
        else:
            output = function(*args)
            cache[args] = output
        return output

    return decorated_function
