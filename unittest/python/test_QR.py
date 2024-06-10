import numpy as np

import eigenpy

rows = 20
cols = 100
rng = np.random.default_rng()

A = rng.random((rows, cols))

# Test HouseholderQR decomposition
householder_qr = eigenpy.HouseholderQR()
householder_qr = eigenpy.HouseholderQR(rows, cols)
householder_qr = eigenpy.HouseholderQR(A)

householder_qr_eye = eigenpy.HouseholderQR(np.eye(rows, rows))
X = rng.random((rows, 20))
assert householder_qr_eye.absDeterminant() == 1.0
assert householder_qr_eye.logAbsDeterminant() == 0.0

Y = householder_qr_eye.solve(X)
assert (X == Y).all()
