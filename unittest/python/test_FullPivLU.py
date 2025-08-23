import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))
fullpivlu = eigenpy.FullPivLU(A)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = fullpivlu.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = fullpivlu.solve(b)
assert eigenpy.is_approx(x, x_est)
assert eigenpy.is_approx(A.dot(x_est), b)

rows = fullpivlu.rows()
cols = fullpivlu.cols()
assert cols == dim
assert rows == dim

fullpivlu_compute = fullpivlu.compute(A)
A_reconstructed = fullpivlu.reconstructedMatrix()
assert eigenpy.is_approx(A_reconstructed, A)

nonzeropivots = fullpivlu.nonzeroPivots()
maxpivot = fullpivlu.maxPivot()
assert nonzeropivots == dim
assert maxpivot > 0

LU = fullpivlu.matrixLU()
P_perm = fullpivlu.permutationP()
Q_perm = fullpivlu.permutationQ()
P = P_perm.toDenseMatrix()
Q = Q_perm.toDenseMatrix()

U = np.triu(LU)
L = np.eye(dim) + np.tril(LU, -1)
assert eigenpy.is_approx(P @ A @ Q, L @ U)

rank = fullpivlu.rank()
dimkernel = fullpivlu.dimensionOfKernel()
injective = fullpivlu.isInjective()
surjective = fullpivlu.isSurjective()
invertible = fullpivlu.isInvertible()
assert rank == dim
assert dimkernel == 0
assert injective
assert surjective
assert invertible

kernel = fullpivlu.kernel()
assert eigenpy.is_approx(A @ kernel, np.zeros((dim, 1)), 1e-10)
image = fullpivlu.image(A)
assert image.shape[1] == rank

inverse = fullpivlu.inverse()
assert eigenpy.is_approx(A @ inverse, np.eye(dim))
assert eigenpy.is_approx(inverse @ A, np.eye(dim))

rcond = fullpivlu.rcond()
determinant = fullpivlu.determinant()
det_numpy = np.linalg.det(A)
assert rcond > 0
assert abs(determinant - det_numpy) / abs(det_numpy) < 1e-10

fullpivlu.setThreshold()
default_threshold = fullpivlu.threshold()
fullpivlu.setThreshold(1e-8)
assert fullpivlu.threshold() == 1e-8

P_inv = P_perm.inverse().toDenseMatrix()
Q_inv = Q_perm.inverse().toDenseMatrix()
assert eigenpy.is_approx(P @ P_inv, np.eye(dim))
assert eigenpy.is_approx(Q @ Q_inv, np.eye(dim))
assert eigenpy.is_approx(P_inv @ P, np.eye(dim))
assert eigenpy.is_approx(Q_inv @ Q, np.eye(dim))

rows_rect = 4
cols_rect = 6
A_rect = rng.random((rows_rect, cols_rect))
fullpivlu_rect = eigenpy.FullPivLU(A_rect)
assert fullpivlu_rect.rows() == rows_rect
assert fullpivlu_rect.cols() == cols_rect
rank_rect = fullpivlu_rect.rank()
assert rank_rect <= min(rows_rect, cols_rect)
assert fullpivlu_rect.dimensionOfKernel() == cols_rect - rank_rect

decomp1 = eigenpy.FullPivLU()
decomp2 = eigenpy.FullPivLU()
id1 = decomp1.id()
id2 = decomp2.id()
assert id1 != id2
assert id1 == decomp1.id()
assert id2 == decomp2.id()

decomp3 = eigenpy.FullPivLU(dim, dim)
decomp4 = eigenpy.FullPivLU(dim, dim)
id3 = decomp3.id()
id4 = decomp4.id()
assert id3 != id4
assert id3 == decomp3.id()
assert id4 == decomp4.id()
