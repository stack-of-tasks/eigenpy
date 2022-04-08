import eigenpy

eigenpy.switchToNumpyArray()

import numpy as np
import numpy.linalg as la

dim = 100
A = np.random.rand(dim, dim)
A = (A + A.T) * 0.5

es = eigenpy.SelfAdjointEigenSolver(A)

V = es.eigenvectors()
D = es.eigenvalues()

assert eigenpy.is_approx(A.dot(V), V.dot(np.diag(D)), 1e-6)
