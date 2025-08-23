import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

es = eigenpy.ComplexEigenSolver(A)
assert es.info() == eigenpy.ComputationInfo.Success

V = es.eigenvectors()
D = es.eigenvalues()
assert V.shape == (dim, dim)
assert D.shape == (dim,)

AV = A @ V
VD = V @ np.diag(D)
assert eigenpy.is_approx(AV.real, VD.real)
assert eigenpy.is_approx(AV.imag, VD.imag)

ces5 = eigenpy.ComplexEigenSolver(A)
ces6 = eigenpy.ComplexEigenSolver(A)
id5 = ces5.id()
id6 = ces6.id()
assert id5 != id6
assert id5 == ces5.id()
assert id6 == ces6.id()
