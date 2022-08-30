import eigenpy
import numpy as np

dim = 100
A = np.eye(dim)

minres = eigenpy.MINRES(A)

X = np.random.rand(dim, 20)
B = A.dot(X)
X_est = minres.solve(B)
print("A.dot(X_est):", A.dot(X_est))
print("B:", B)
assert eigenpy.is_approx(A.dot(X_est), B, 1e-6)
