import numpy as np
from scipy.sparse import csc_matrix

import eigenpy

rng = np.random.default_rng()


def test(SolverType: type):
    dim = 100
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    A = csc_matrix(A)

    llt = SolverType(A)

    assert llt.info() == eigenpy.ComputationInfo.Success

    X = rng.random((dim, 20))
    B = A.dot(X)
    X_est = llt.solve(B)
    #    import pdb; pdb.set_trace()
    assert eigenpy.is_approx(X, X_est)
    assert eigenpy.is_approx(A.dot(X_est), B)

    llt.analyzePattern(A)
    llt.factorize(A)


test(eigenpy.AccelerateLLT)
test(eigenpy.AccelerateLDLT)
test(eigenpy.AccelerateLDLTUnpivoted)
test(eigenpy.AccelerateLDLTSBK)
test(eigenpy.AccelerateLDLTTPP)
test(eigenpy.AccelerateQR)
# test(eigenpy.AccelerateCholeskyAtA) # This test is not passing. Seems there is a bug in Eigen with the support of Accelerate.
