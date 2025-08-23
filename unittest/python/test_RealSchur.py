import numpy as np

import eigenpy


def verify_is_quasi_triangular(T):
    size = T.shape[0]

    for row in range(2, size):
        for col in range(row - 1):
            assert abs(T[row, col]) < 1e-12

    for row in range(1, size):
        if abs(T[row, row - 1]) > 1e-12:
            if row < size - 1:
                assert abs(T[row + 1, row]) < 1e-12

            tr = T[row - 1, row - 1] + T[row, row]
            det = T[row - 1, row - 1] * T[row, row] - T[row - 1, row] * T[row, row - 1]
            assert 4 * det > tr * tr


dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

rs = eigenpy.RealSchur(A)
assert rs.info() == eigenpy.ComputationInfo.Success

U = rs.matrixU()
T = rs.matrixT()

assert eigenpy.is_approx(A, U @ T @ U.T)
assert eigenpy.is_approx(U @ U.T, np.eye(dim))

verify_is_quasi_triangular(T)

hess = eigenpy.HessenbergDecomposition(A)
H = hess.matrixH()
Q_hess = hess.matrixQ()

rs_from_hess = eigenpy.RealSchur(dim)
result_from_hess = rs_from_hess.computeFromHessenberg(H, Q_hess, True)
assert result_from_hess.info() == eigenpy.ComputationInfo.Success

T_from_hess = rs_from_hess.matrixT()
U_from_hess = rs_from_hess.matrixU()

assert eigenpy.is_approx(A, U_from_hess @ T_from_hess @ U_from_hess.T)
