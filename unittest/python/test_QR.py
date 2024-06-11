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

# Test FullPivHouseholderQR decomposition
fullpiv_householder_qr = eigenpy.FullPivHouseholderQR()
fullpiv_householder_qr = eigenpy.FullPivHouseholderQR(rows, cols)
fullpiv_householder_qr = eigenpy.FullPivHouseholderQR(A)

fullpiv_householder_qr = eigenpy.FullPivHouseholderQR(np.eye(rows, rows))
X = rng.random((rows, 20))
assert fullpiv_householder_qr.absDeterminant() == 1.0
assert fullpiv_householder_qr.logAbsDeterminant() == 0.0

Y = fullpiv_householder_qr.solve(X)
assert (X == Y).all()
assert fullpiv_householder_qr.rank() == rows

fullpiv_householder_qr.setThreshold(1e-8)
assert fullpiv_householder_qr.threshold() == 1e-8
assert eigenpy.is_approx(np.eye(rows, rows), fullpiv_householder_qr.inverse())

# Test ColPivHouseholderQR decomposition
colpiv_householder_qr = eigenpy.ColPivHouseholderQR()
colpiv_householder_qr = eigenpy.ColPivHouseholderQR(rows, cols)
colpiv_householder_qr = eigenpy.ColPivHouseholderQR(A)

colpiv_householder_qr = eigenpy.ColPivHouseholderQR(np.eye(rows, rows))
X = rng.random((rows, 20))
assert colpiv_householder_qr.absDeterminant() == 1.0
assert colpiv_householder_qr.logAbsDeterminant() == 0.0

Y = colpiv_householder_qr.solve(X)
assert (X == Y).all()
assert colpiv_householder_qr.rank() == rows

colpiv_householder_qr.setThreshold(1e-8)
assert colpiv_householder_qr.threshold() == 1e-8
assert eigenpy.is_approx(np.eye(rows, rows), colpiv_householder_qr.inverse())

# Test CompleteOrthogonalDecomposition
cod = eigenpy.CompleteOrthogonalDecomposition()
cod = eigenpy.CompleteOrthogonalDecomposition(rows, cols)
cod = eigenpy.CompleteOrthogonalDecomposition(A)

cod = eigenpy.CompleteOrthogonalDecomposition(np.eye(rows, rows))
X = rng.random((rows, 20))
assert cod.absDeterminant() == 1.0
assert cod.logAbsDeterminant() == 0.0

Y = cod.solve(X)
assert (X == Y).all()
assert cod.rank() == rows

cod.setThreshold(1e-8)
assert cod.threshold() == 1e-8
assert eigenpy.is_approx(np.eye(rows, rows), cod.pseudoInverse())
