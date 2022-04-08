import numpy as np
from eigen_ref import *


def test(mat):

    printMatrix(mat)
    fill(mat, 1.0)
    printMatrix(mat)
    assert np.array_equal(mat, np.full(mat.shape, 1.0))

    A_ref = asRef(mat.shape[0], mat.shape[1])
    A_ref.fill(1.0)
    A_ref2 = asRef(mat.shape[0], mat.shape[1])

    assert np.array_equal(A_ref, A_ref2)

    A_ref2.fill(0)
    assert np.array_equal(A_ref, A_ref2)

    ref = asRef(mat)
    assert np.all(ref == mat)

    const_ref = asConstRef(mat)
    assert np.all(const_ref == mat)


rows = 10
cols = 30

mat = np.ones((rows, cols), order="F")

test(mat)
