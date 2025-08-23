import numpy as np

import eigenpy

_options = [
    eigenpy.DecompositionOptions.ComputeEigenvectors
    | eigenpy.DecompositionOptions.Ax_lBx,
    eigenpy.DecompositionOptions.EigenvaluesOnly | eigenpy.DecompositionOptions.Ax_lBx,
    eigenpy.DecompositionOptions.ComputeEigenvectors
    | eigenpy.DecompositionOptions.ABx_lx,
    eigenpy.DecompositionOptions.EigenvaluesOnly | eigenpy.DecompositionOptions.ABx_lx,
    eigenpy.DecompositionOptions.ComputeEigenvectors
    | eigenpy.DecompositionOptions.BAx_lx,
    eigenpy.DecompositionOptions.EigenvaluesOnly | eigenpy.DecompositionOptions.BAx_lx,
]


def test_generalized_selfadjoint_eigensolver(options):
    dim = 100
    rng = np.random.default_rng()
    A = rng.random((dim, dim))
    A = (A + A.T) * 0.5
    B = rng.random((dim, dim))
    B = B @ B.T + 0.1 * np.eye(dim)

    gsaes = eigenpy.GeneralizedSelfAdjointEigenSolver(A, B, options)
    assert gsaes.info() == eigenpy.ComputationInfo.Success

    D = gsaes.eigenvalues()
    assert D.shape == (dim,)
    assert all(abs(D[i].imag) < 1e-12 for i in range(dim))
    assert all(D[i] <= D[i + 1] + 1e-12 for i in range(dim - 1))

    compute_eigenvectors = bool(
        options & eigenpy.DecompositionOptions.ComputeEigenvectors
    )

    if compute_eigenvectors:
        V = gsaes.eigenvectors()
        assert V.shape == (dim, dim)

        if options & eigenpy.DecompositionOptions.Ax_lBx:
            for i in range(dim):
                v = V[:, i]
                lam = D[i]
                Av = A @ v
                lam_Bv = lam * (B @ v)
                assert eigenpy.is_approx(Av, lam_Bv, 1e-6)

            VT_B_V = V.T @ B @ V
            assert eigenpy.is_approx(VT_B_V, np.eye(dim), 1e-6)

        elif options & eigenpy.DecompositionOptions.ABx_lx:
            AB = A @ B
            for i in range(dim):
                v = V[:, i]
                lam = D[i]
                ABv = AB @ v
                lam_v = lam * v
                assert eigenpy.is_approx(ABv, lam_v, 1e-6)

        elif options & eigenpy.DecompositionOptions.BAx_lx:
            BA = B @ A
            for i in range(dim):
                v = V[:, i]
                lam = D[i]
                BAv = BA @ v
                lam_v = lam * v
                assert eigenpy.is_approx(BAv, lam_v, 1e-6)

    _gsaes_compute = gsaes.compute(A, B)
    _gsaes_compute_options = gsaes.compute(A, B, options)

    rank = len([d for d in D if abs(d) > 1e-12])
    assert rank <= dim


for opt in _options:
    test_generalized_selfadjoint_eigensolver(opt)
