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
    # import pdb; pdb.set_trace()
    assert np.all(const_ref == mat)

    mat.fill(0.0)
    fill(mat[:3, :2], 1.0)

    assert np.all(mat[:3, :2] == np.ones((3, 2)))

    mat.fill(0.0)
    fill(mat[:2, :3], 1.0)

    assert np.all(mat[:2, :3] == np.ones((2, 3)))

    mat.fill(0.0)
    mat_as_C_order = np.array(mat, order="F")
    getBlock(mat_as_C_order, 0, 0, 3, 2)[:, :] = 1.0

    assert np.all(mat_as_C_order[:3, :2] == np.ones((3, 2)))

    class ModifyBlockImpl(modify_block):
        def __init__(self):
            super().__init__()

        def call(self, mat):
            mat[:, :] = 1.0

    modify = ModifyBlockImpl()
    modify.modify(2, 3)
    Jref = np.zeros((10, 10))
    Jref[:2, :3] = 1.0

    assert np.array_equal(Jref, modify.J)


rows = 10
cols = 30

mat = np.ones((rows, cols), order="F")

test(mat)
