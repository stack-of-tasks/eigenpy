import eigenpy

import numpy as np
import numpy.linalg as la

dim = 100
A = np.random.rand(dim,dim)

A = (A + A.T)*0.5 + np.diag(10. + np.random.rand(dim))

minres = eigenpy.MINRES(A)

X = np.random.rand(dim,20)
B = A.dot(X)
X_est = minres.solve(B)
assert eigenpy.is_approx(X,X_est)
assert eigenpy.is_approx(A.dot(X_est),B)
