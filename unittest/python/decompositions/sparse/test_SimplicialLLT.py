import numpy as np
from scipy.sparse import csc_matrix

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

A = csc_matrix(A)

llt = eigenpy.SimplicialLLT(A)

assert llt.info() == eigenpy.ComputationInfo.Success

L = llt.matrixL()
U = llt.matrixU()

LU = L @ U
assert eigenpy.is_approx(LU.toarray(), A.toarray())

X = rng.random((dim, 20))
B = A.dot(X)
X_est = llt.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

llt.analyzePattern(A)
llt.factorize(A)
permutation = llt.permutationP()
