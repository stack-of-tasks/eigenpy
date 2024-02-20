import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5

es = eigenpy.SelfAdjointEigenSolver(A)

V = es.eigenvectors()
D = es.eigenvalues()

assert eigenpy.is_approx(A.dot(V), V.dot(np.diag(D)), 1e-6)
