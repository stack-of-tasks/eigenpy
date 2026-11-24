import numpy as np
from scipy.sparse import csc_matrix

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(5.0 + rng.random(dim))
A = csc_matrix(A)

ichol = eigenpy.solvers.IncompleteCholesky(A)
assert ichol.info() == eigenpy.ComputationInfo.Success
assert ichol.rows() == dim
assert ichol.cols() == dim

X = rng.random((dim, 20))
B = A.dot(X)
X_est = ichol.solve(B)
assert isinstance(X_est, np.ndarray)
residual = np.linalg.norm(B - A.dot(X_est)) / np.linalg.norm(B)
assert residual < 0.1

x = rng.random(dim)
b = A.dot(x)
x_est = ichol.solve(b)
assert isinstance(x_est, np.ndarray)
residual = np.linalg.norm(b - A.dot(x_est)) / np.linalg.norm(b)
assert residual < 0.1

X_sparse = csc_matrix(rng.random((dim, 10)))
B_sparse = A.dot(X_sparse).tocsc()
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()
X_est_sparse = ichol.solve(B_sparse)
assert isinstance(X_est_sparse, csc_matrix)

ichol.analyzePattern(A)
ichol.factorize(A)
ichol.compute(A)
assert ichol.info() == eigenpy.ComputationInfo.Success

L = ichol.matrixL()
S_diag = ichol.scalingS()
perm = ichol.permutationP()
P = perm.toDenseMatrix()

assert isinstance(L, csc_matrix)
assert isinstance(S_diag, np.ndarray)
assert L.shape == (dim, dim)
assert S_diag.shape == (dim,)

L_dense = L.toarray()
upper_part = np.triu(L_dense, k=1)
assert np.allclose(upper_part, 0, atol=1e-12)

assert np.all(S_diag > 0)

S = csc_matrix((S_diag, (range(dim), range(dim))), shape=(dim, dim))

PA = P @ A
PAP = PA @ P.T
SPAP = S @ PAP
SPAPS = SPAP @ S

LLT = L @ L.T

diff = SPAPS - LLT
relative_error = np.linalg.norm(diff.data) / np.linalg.norm(SPAPS.data)
assert relative_error < 0.5
