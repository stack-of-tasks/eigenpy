import numpy as np
import eigenpy

eigenpy.switchToNumpyArray()

dim = 100
A = np.random.rand(dim, dim)

es = eigenpy.EigenSolver(A)

V = es.eigenvectors()
D = es.eigenvalues()

assert eigenpy.is_approx(A.dot(V).real, V.dot(np.diag(D)).real)
assert eigenpy.is_approx(A.dot(V).imag, V.dot(np.diag(D)).imag)
