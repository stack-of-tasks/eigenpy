import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))

realqz = eigenpy.RealQZ(A, B)
assert realqz.info() == eigenpy.ComputationInfo.Success

Q = realqz.matrixQ()
S = realqz.matrixS()
Z = realqz.matrixZ()
T = realqz.matrixT()

assert eigenpy.is_approx(A, Q @ S @ Z)
assert eigenpy.is_approx(B, Q @ T @ Z)

assert eigenpy.is_approx(Q @ Q.T, np.eye(dim))
assert eigenpy.is_approx(Z @ Z.T, np.eye(dim))

for i in range(dim):
    for j in range(i):
        assert abs(T[i, j]) < 1e-12

for i in range(dim):
    for j in range(i - 1):
        assert abs(S[i, j]) < 1e-12

realqz3_id = eigenpy.RealQZ(A, B)
realqz4_id = eigenpy.RealQZ(A, B)
id3 = realqz3_id.id()
id4 = realqz4_id.id()
assert id3 != id4
assert id3 == realqz3_id.id()
assert id4 == realqz4_id.id()
