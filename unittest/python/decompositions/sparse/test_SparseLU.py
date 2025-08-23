import numpy as np
import scipy.sparse as spa

import eigenpy

dim = 100
rng = np.random.default_rng()

A_fac = spa.random(dim, dim, density=0.25, random_state=rng)
A = A_fac.T @ A_fac
A += spa.diags(10.0 * rng.standard_normal(dim) ** 2)
A = A.tocsc(True)
A.check_format()

splu = eigenpy.SparseLU(A)

assert splu.info() == eigenpy.ComputationInfo.Success

X = rng.random((dim, 20))
B = A.dot(X)
X_est = splu.solve(B)
assert isinstance(X_est, np.ndarray)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

splu.analyzePattern(A)
splu.factorize(A)

X_sparse = spa.random(dim, 10, random_state=rng)
B_sparse = A.dot(X_sparse)
B_sparse: spa.csc_matrix = B_sparse.tocsc(True)
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()

X_est = splu.solve(B_sparse)
assert isinstance(X_est, spa.csc_matrix)
assert eigenpy.is_approx(X_est.toarray(), X_sparse.toarray())
assert eigenpy.is_approx(A.dot(X_est.toarray()), B_sparse.toarray())

assert splu.nnzL() > 0
assert splu.nnzU() > 0

L = splu.matrixL()
U = splu.matrixU()

assert L.rows() == dim
assert L.cols() == dim
assert U.rows() == dim
assert U.cols() == dim

x_true = rng.random(dim)
b_true = A.dot(x_true)
P_rows_indices = splu.rowsPermutation().indices()
P_cols_indices = splu.colsPermutation().indices()

b_permuted = b_true[P_rows_indices]
z = b_permuted.copy()
L.solveInPlace(z)
y = z.copy()
U.solveInPlace(y)
x_reconstructed = np.zeros(dim)
x_reconstructed[P_cols_indices] = y

assert eigenpy.is_approx(x_reconstructed, x_true, prec=1e-6)
