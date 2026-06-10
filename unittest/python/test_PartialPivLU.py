import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))
partialpivlu = eigenpy.PartialPivLU(A)

X = rng.random((dim, 20))
B = A.dot(X)
X_est = partialpivlu.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

x = rng.random(dim)
b = A.dot(x)
x_est = partialpivlu.solve(b)
assert eigenpy.is_approx(x, x_est)
assert eigenpy.is_approx(A.dot(x_est), b)

rows = partialpivlu.rows()
cols = partialpivlu.cols()
assert cols == dim
assert rows == dim

partialpivlu_compute = partialpivlu.compute(A)
A_reconstructed = partialpivlu.reconstructedMatrix()
assert eigenpy.is_approx(A_reconstructed, A)

LU = partialpivlu.matrixLU()
P_perm = partialpivlu.permutationP()
P = P_perm.toDenseMatrix()

U = np.triu(LU)
L = np.eye(dim) + np.tril(LU, -1)
assert eigenpy.is_approx(P @ A, L @ U)

inverse = partialpivlu.inverse()
assert eigenpy.is_approx(A @ inverse, np.eye(dim))
assert eigenpy.is_approx(inverse @ A, np.eye(dim))

rcond = partialpivlu.rcond()
determinant = partialpivlu.determinant()
det_numpy = np.linalg.det(A)
assert rcond > 0
assert abs(determinant - det_numpy) / abs(det_numpy) < 1e-10

P_inv = P_perm.inverse().toDenseMatrix()
assert eigenpy.is_approx(P @ P_inv, np.eye(dim))
assert eigenpy.is_approx(P_inv @ P, np.eye(dim))

decomp1 = eigenpy.PartialPivLU()
decomp2 = eigenpy.PartialPivLU()
id1 = decomp1.id()
id2 = decomp2.id()
assert id1 != id2
assert id1 == decomp1.id()
assert id2 == decomp2.id()

decomp3 = eigenpy.PartialPivLU(dim)
decomp4 = eigenpy.PartialPivLU(dim)
id3 = decomp3.id()
id4 = decomp4.id()
assert id3 != id4
assert id3 == decomp3.id()
assert id4 == decomp4.id()

decomp5 = eigenpy.PartialPivLU(A)
decomp6 = eigenpy.PartialPivLU(A)
id5 = decomp5.id()
id6 = decomp6.id()
assert id5 != id6
assert id5 == decomp5.id()
assert id6 == decomp6.id()
