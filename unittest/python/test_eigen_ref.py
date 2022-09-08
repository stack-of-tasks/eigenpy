import numpy as np
from eigen_ref import (
    printMatrix,
    asRef,
    asConstRef,
    fill,
    getBlock,
    editBlock,
    modify_block,
    has_ref_member,
)


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

    mat.fill(0.0)
    fill(mat[:3, :2], 1.0)

    assert np.all(mat[:3, :2] == np.ones((3, 2)))

    mat.fill(0.0)
    fill(mat[:2, :3], 1.0)

    assert np.all(mat[:2, :3] == np.ones((2, 3)))

    mat.fill(0.0)
    mat[:, :] = np.arange(rows * cols).reshape(rows, cols)
    printMatrix(mat)
    mat0 = mat.copy()
    mat_as_C_order = np.array(mat, order="F")
    for i, rowsize, colsize in ([0, 3, 2], [1, 1, 2], [0, 3, 1]):
        print("taking block [{}:{}, {}:{}]".format(i, rowsize + i, 0, colsize))
        B = getBlock(mat_as_C_order, i, 0, rowsize, colsize)
        lhs = mat_as_C_order[i : rowsize + i, :colsize]
        print("should be:\n{}\ngot:\n{}".format(lhs, B))
        assert np.array_equal(lhs, B.reshape(rowsize, colsize))

        B[:] = 1.0
        rhs = np.ones((rowsize, colsize))
        assert np.array_equal(mat_as_C_order[i : rowsize + i, :colsize], rhs)

        mat_as_C_order[:, :] = mat0

    mat_copy = mat_as_C_order.copy()
    editBlock(mat_as_C_order, 0, 0, 3, 2)
    mat_copy[:3, :2] = np.arange(6).reshape(3, 2)

    assert np.array_equal(mat_as_C_order, mat_copy)

    class ModifyBlockImpl(modify_block):
        def __init__(self):
            super(ModifyBlockImpl, self).__init__()

        def call(self, mat):
            n, m = mat.shape
            mat[:, :] = np.arange(n * m).reshape(n, m)

    modify = ModifyBlockImpl()
    modify.modify(2, 3)
    Jref = np.zeros((10, 10))
    Jref[:2, :3] = np.arange(6).reshape(2, 3)

    assert np.array_equal(Jref, modify.J)

    hasref = has_ref_member()
    A = np.ones((3, 3)) / 2
    hasref.Jref[:, :] = A
    J_true = np.zeros((4, 4))
    J_true[:3, 1:] = A

    assert np.array_equal(hasref.J, J_true)


rows = 10
cols = 30

mat = np.ones((rows, cols), order="F")

test(mat)
