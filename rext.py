import numpy as np
import rpy2.robjects as robjects


def ToR(pyobj):
    r = robjects.r

    if isinstance(pyobj, list) or isinstance(pyobj, tuple):
        pyobj = np.array(pyobj)

    if isinstance(pyobj, np.ndarray):
        if pyobj.ndim == 1:
            if pyobj.dtype == float:
                return robjects.FloatVector(pyobj)
            if pyobj.dtype == int:
                return robjects.IntVector(pyobj)
        if pyobj.ndim == 2:
            if pyobj.dtype == float:
                return r.matrix(robjects.FloatVector(pyobj.T.flatten()), pyobj.shape[0])
            if pyobj.dtype == int:
                return r.matrix(robjects.IntVector(pyobj.T.flatten()), pyobj.shape[0])

    raise StandardError("ToR: %s is not implemented" % type(pyobj))


def FromR(robj):
    if isinstance(robj, robjects.vectors.Matrix):
        return np.array(robj)
    if isinstance(robj, robjects.FloatVector):
        return np.array(robj)
    raise StandardError("FromR: %s is not implemented" % type(robj))


class MultivariateKde(object):

    def __init__(self, data):
        from rpy2.robjects.packages import importr
        self.ks = importr('ks')
        self.ndim = data.shape[1]
        self.data = ToR(data)
        self.hmatrix = self.ks.Hpi(x=self.data)

    def __call__(self, vs):
        vs = np.atleast_2d(vs)
        if vs.shape[1] != self.ndim:
            raise StandardError("dimensions don't match")
        robjects.globalenv["vs"] = ToR(vs)
        robjects.globalenv["data"] = self.data
        robjects.globalenv["hmatrix"] = self.hmatrix
        rout = robjects.r("kde(x=data,H=hmatrix,eval.points=vs)")[2]
        return FromR(rout)
