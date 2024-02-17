import numpy as np
from scipy.sparse import csc_matrix

import eigenpy

dim = 100
A = np.random.rand(dim, dim)
A = (A + A.T) * 0.5 + np.diag(10.0 + np.random.rand(dim))

A = csc_matrix(A)

ldlt = eigenpy.SimplicialLDLT(A)

assert ldlt.info() == eigenpy.ComputationInfo.Success

L = ldlt.matrixL()
U = ldlt.matrixU()
D = csc_matrix(np.diag(ldlt.vectorD()))

LDU = L @ D @ U
assert eigenpy.is_approx(LDU.toarray(), A.toarray())

X = np.random.rand(dim, 20)
B = A.dot(X)
X_est = ldlt.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

ldlt.analyzePattern(A)
ldlt.factorize(A)
permutation = ldlt.permutationP()
