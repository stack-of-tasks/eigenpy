import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

cs = eigenpy.ComplexSchur(A)
assert cs.info() == eigenpy.ComputationInfo.Success

U = cs.matrixU()
T = cs.matrixT()

A_complex = A.astype(complex)
assert eigenpy.is_approx(A_complex, U @ T @ U.conj().T)
assert eigenpy.is_approx(U @ U.conj().T, np.eye(dim))

for row in range(1, dim):
    for col in range(row):
        assert abs(T[row, col]) < 1e-12

A_triangular = np.triu(A)
cs_triangular = eigenpy.ComplexSchur(dim)
cs_triangular.setMaxIterations(1)
result_triangular = cs_triangular.compute(A_triangular)
assert result_triangular.info() == eigenpy.ComputationInfo.Success

T_triangular = cs_triangular.matrixT()
U_triangular = cs_triangular.matrixU()

A_triangular_complex = A_triangular.astype(complex)
assert eigenpy.is_approx(T_triangular, A_triangular_complex)
assert eigenpy.is_approx(U_triangular, np.eye(dim, dtype=complex))

hess = eigenpy.HessenbergDecomposition(A)
H = hess.matrixH()
Q_hess = hess.matrixQ()

cs_from_hess = eigenpy.ComplexSchur(dim)
result_from_hess = cs_from_hess.computeFromHessenberg(H, Q_hess, True)
assert result_from_hess.info() == eigenpy.ComputationInfo.Success

T_from_hess = cs_from_hess.matrixT()
U_from_hess = cs_from_hess.matrixU()

A_complex = A.astype(complex)
assert eigenpy.is_approx(A_complex, U_from_hess @ T_from_hess @ U_from_hess.conj().T)
