import numpy as np
from scipy.sparse import csc_matrix

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(5.0 + rng.random(dim))
A = csc_matrix(A)

ilut = eigenpy.solvers.IncompleteLUT(A)
assert ilut.info() == eigenpy.ComputationInfo.Success
assert ilut.rows() == dim
assert ilut.cols() == dim

X = rng.random((dim, 100))
B = A.dot(X)
X_est = ilut.solve(B)
assert isinstance(X_est, np.ndarray)
residual = np.linalg.norm(B - A.dot(X_est)) / np.linalg.norm(B)
assert residual < 0.1

x = rng.random(dim)
b = A.dot(x)
x_est = ilut.solve(b)
assert isinstance(x_est, np.ndarray)
residual = np.linalg.norm(b - A.dot(x_est)) / np.linalg.norm(b)
assert residual < 0.1

X_sparse = csc_matrix(rng.random((dim, 10)))
B_sparse = A.dot(X_sparse).tocsc()
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()
X_est_sparse = ilut.solve(B_sparse)
assert isinstance(X_est_sparse, csc_matrix)

ilut.analyzePattern(A)
ilut.factorize(A)
assert ilut.info() == eigenpy.ComputationInfo.Success

ilut_params = eigenpy.solvers.IncompleteLUT(A, 1e-4, 15)
assert ilut_params.info() == eigenpy.ComputationInfo.Success

ilut_set = eigenpy.solvers.IncompleteLUT()
ilut_set.setDroptol(1e-3)
ilut_set.setFillfactor(20)
ilut_set.compute(A)
assert ilut_set.info() == eigenpy.ComputationInfo.Success
