import eigenpy

import numpy as np
import numpy.linalg as la

dim = 100
A = np.random.rand(dim, dim)

A = (A + A.T) * 0.5 + np.diag(10.0 + np.random.rand(dim))

llt = eigenpy.LLT(A)

L = llt.matrixL()
assert eigenpy.is_approx(L.dot(np.transpose(L)), A)

X = np.random.rand(dim, 20)
B = A.dot(X)
X_est = llt.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)
