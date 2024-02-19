import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

ldlt = eigenpy.LDLT(A)

L = ldlt.matrixL()
D = ldlt.vectorD()
P = ldlt.transpositionsP()

assert eigenpy.is_approx(
    np.transpose(P).dot(L.dot(np.diag(D).dot(np.transpose(L).dot(P)))), A
)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = ldlt.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)
