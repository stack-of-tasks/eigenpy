import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))
B = (B + B.T) * 0.5 + np.diag(10.0 + rng.random(dim))

ges_matrices = eigenpy.GeneralizedEigenSolver(A, B)
assert ges_matrices.info() == eigenpy.ComputationInfo.Success

alphas = ges_matrices.alphas()
betas = ges_matrices.betas()
eigenvectors = ges_matrices.eigenvectors()
eigenvalues = ges_matrices.eigenvalues()

for k in range(dim):
    v = eigenvectors[:, k]
    lambda_k = eigenvalues[k]

    Av = A @ v
    lambda_Bv = lambda_k * (B @ v)
    assert eigenpy.is_approx(Av.real, lambda_Bv.real, 1e-6)
    assert eigenpy.is_approx(Av.imag, lambda_Bv.imag, 1e-6)

for k in range(dim):
    v = eigenvectors[:, k]
    alpha = alphas[k]
    beta = betas[k]

    alpha_Bv = alpha * (B @ v)
    beta_Av = beta * (A @ v)
    assert eigenpy.is_approx(alpha_Bv.real, beta_Av.real, 1e-6)
    assert eigenpy.is_approx(alpha_Bv.imag, beta_Av.imag, 1e-6)

for k in range(dim):
    if abs(betas[k]) > 1e-12:
        expected_eigenvalue = alphas[k] / betas[k]
        assert abs(eigenvalues[k] - expected_eigenvalue) < 1e-12
