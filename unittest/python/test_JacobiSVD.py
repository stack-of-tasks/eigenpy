import numpy as np

import eigenpy

THIN_U = eigenpy.DecompositionOptions.ComputeThinU
THIN_V = eigenpy.DecompositionOptions.ComputeThinV
FULL_U = eigenpy.DecompositionOptions.ComputeFullU
FULL_V = eigenpy.DecompositionOptions.ComputeFullV

_options = [
    0,
    # THIN_U,
    # THIN_V,
    # FULL_U,
    # FULL_V,
    THIN_U | THIN_V,
    FULL_U | FULL_V,
    # THIN_U | FULL_V,
    # FULL_U | THIN_V,
]

_classes = [
    eigenpy.ColPivHhJacobiSVD,
    # eigenpy.FullPivHhJacobiSVD,
    # eigenpy.HhJacobiSVD,
    # eigenpy.NoPrecondJacobiSVD,
]

# Rationale: Tets only few cases to gain computation time
# User can test all of them by uncommenting the corresponding lines


def is_valid_combination(cls, opt):
    if cls == eigenpy.FullPivHhJacobiSVD:
        has_thin_u = bool(opt & THIN_U)
        has_thin_v = bool(opt & THIN_V)

        if has_thin_u or has_thin_v:
            return False

    return True


def test_jacobi(cls, opt):
    dim = 100
    rng = np.random.default_rng()
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

    jacobisvd = cls(A, opt)
    assert jacobisvd.info() == eigenpy.ComputationInfo.Success

    has_u = opt & (THIN_U | FULL_U)
    has_v = opt & (THIN_V | FULL_V)

    if has_u and has_v:
        X = rng.random((dim, 20))
        B = A @ X
        X_est = jacobisvd.solve(B)
        assert eigenpy.is_approx(X, X_est)
        assert eigenpy.is_approx(A @ X_est, B)

        x = rng.random(dim)
        b = A @ x
        x_est = jacobisvd.solve(b)
        assert eigenpy.is_approx(x, x_est)
        assert eigenpy.is_approx(A @ x_est, b)

    assert jacobisvd.rows() == dim
    assert jacobisvd.cols() == dim

    _jacobisvd_compute = jacobisvd.compute(A)
    _jacobisvd_compute_options = jacobisvd.compute(A, opt)

    rank = jacobisvd.rank()
    singularvalues = jacobisvd.singularValues()
    nonzerosingularvalues = jacobisvd.nonzeroSingularValues()
    assert rank == nonzerosingularvalues
    assert len(singularvalues) == dim
    assert all(
        singularvalues[i] >= singularvalues[i + 1]
        for i in range(len(singularvalues) - 1)
    )

    compute_u = jacobisvd.computeU()
    compute_v = jacobisvd.computeV()
    expected_compute_u = bool(has_u)
    expected_compute_v = bool(has_v)
    assert compute_u == expected_compute_u
    assert compute_v == expected_compute_v

    if compute_u:
        matrixU = jacobisvd.matrixU()
        assert matrixU.shape == (dim, dim)
        assert eigenpy.is_approx(matrixU.T @ matrixU, np.eye(matrixU.shape[1]))

    if compute_v:
        matrixV = jacobisvd.matrixV()
        assert matrixV.shape == (dim, dim)
        assert eigenpy.is_approx(matrixV.T @ matrixV, np.eye(matrixV.shape[1]))

    if compute_u and compute_v:
        U = jacobisvd.matrixU()
        V = jacobisvd.matrixV()
        S = jacobisvd.singularValues()
        S_matrix = np.diag(S)
        A_reconstructed = U @ S_matrix @ V.T
        assert eigenpy.is_approx(A, A_reconstructed)


for cls in _classes:
    for opt in _options:
        if is_valid_combination(cls, opt):
            test_jacobi(cls, opt)
