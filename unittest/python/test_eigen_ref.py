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


def test_fill_print(mat):

    print("print matrix:")
    printMatrix(mat)
    print("calling fill():")
    fill(mat, 1.0)
    print("print again:")
    printMatrix(mat)
    assert np.array_equal(mat, np.full(mat.shape, 1.0))


def test_create_ref_to_static(mat):
    # create ref to static:
    print()
    print("[asRef(int, int)]")
    A_ref = asRef(mat.shape[0], mat.shape[1])
    A_ref.fill(1.0)
    A_ref[0, 1] = -1.0
    print("make second reference:")
    A_ref2 = asRef(mat.shape[0], mat.shape[1])
    print(A_ref2)

    assert np.array_equal(A_ref, A_ref2)

    A_ref2.fill(0)
    assert np.array_equal(A_ref, A_ref2)


def test_create_ref(mat):
    # create ref to input:

    print("[asRef(mat)]")
    ref = asRef(mat)
    assert np.array_equal(ref, mat)
    assert not (ref.flags.owndata)
    assert ref.flags.writeable

    print("[asConstRef]")
    const_ref = asConstRef(mat)
    print(const_ref.flags)
    assert np.array_equal(const_ref, mat)
    assert not (const_ref.flags.writeable)
    assert not (const_ref.flags.owndata)

    print("fill a slice")
    mat[:, :] = 0.0
    fill(mat[:3, :2], 1.0)
    assert np.array_equal(mat[:3, :2], np.ones((3, 2)))

    mat[:, :] = 0.0
    fill(mat[:2, :3], 1.0)
    assert np.array_equal(mat[:2, :3], np.ones((2, 3)))

    print("set mat data to arange()")
    mat.fill(0.0)
    mat[:, :] = np.arange(rows * cols).reshape(rows, cols)
    mat0 = mat.copy()
    mat_as_C_order = np.array(mat, order="F")
    for i, rowsize, colsize in ([0, 3, 2], [1, 1, 2], [0, 3, 1]):
        print("taking block [{}:{}, {}:{}]".format(i, rowsize + i, 0, colsize))
        B = getBlock(mat_as_C_order, i, 0, rowsize, colsize)
        lhs = mat_as_C_order[i : rowsize + i, :colsize]
        assert np.array_equal(lhs, B.reshape(rowsize, colsize))

        B[:] = 1.0
        rhs = np.ones((rowsize, colsize))
        assert np.array_equal(mat_as_C_order[i : rowsize + i, :colsize], rhs)

        mat_as_C_order[:, :] = mat0

    mat_copy = mat_as_C_order.copy()
    print("[editBlock]")
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


def _do_test(mat):
    test_fill_print(mat)
    test_create_ref_to_static(mat)
    test_create_ref(mat)


if __name__ == "__main__":
    rows = 8
    cols = 10

    mat = np.ones((rows, cols), order="F")
    mat[0, 0] = 0
    mat[1:5, 1:5] = 6
    _do_test(mat)

    mat = np.ones((rows, cols))
    mat[2:4, 1:4] = 2
    _do_test(mat)
    mat_f = np.asfortranarray(mat)
    _do_test(mat_f)
