import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

es = eigenpy.EigenSolver(A)

V = es.eigenvectors()
D = es.eigenvalues()

assert eigenpy.is_approx(A.dot(V).real, V.dot(np.diag(D)).real)
assert eigenpy.is_approx(A.dot(V).imag, V.dot(np.diag(D)).imag)
