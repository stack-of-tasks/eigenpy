import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = np.eye(dim)

minres = eigenpy.solvers.MINRES(A)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = minres.solve(B)
assert eigenpy.is_approx(A.dot(X_est), B, 1e-6)

minres_back = eigenpy.MINRES(A)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = minres_back.solve(B)
assert eigenpy.is_approx(A.dot(X_est), B, 1e-6)
