import numpy as np
from eigen_ref import (
    asConstRef,
    asRef,
    copyRowVectorFromConstRef,
    copyVectorFromConstRef,
    editBlock,
    fill,
    getBlock,
    getRefToStatic,
    has_ref_member,
    modify_block,
    printMatrix,
)


def test_fill_print(mat):
    print("print matrix:")
    printMatrix(mat)
    print("calling fill():")
    fill(mat, 1.0)
    assert np.array_equal(mat, np.full(mat.shape, 1.0))

    print("fill a slice")
    mat[:, :] = 0.0
    fill(mat[:3, :2], 1.0)
    printMatrix(mat[:3, :2])
    assert np.array_equal(mat[:3, :2], np.ones((3, 2)))


def test_create_ref_to_static(mat):
    # create ref to static:
    print()
    print("[asRef(int, int)]")
    A_ref = getRefToStatic(mat.shape[0], mat.shape[1])
    A_ref.fill(1.0)
    A_ref[0, 1] = -1.0
    print("make second reference:")
    A_ref2 = getRefToStatic(mat.shape[0], mat.shape[1])
    print(A_ref2)

    assert np.array_equal(A_ref, A_ref2)

    A_ref2.fill(0)
    assert np.array_equal(A_ref, A_ref2)


def test_read_block():
    data = np.array([[0, 0.2, 0.3, 0.4], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])

    data_strided = data[:, 0]

    data_strided_copy = copyVectorFromConstRef(data_strided)
    assert np.all(data_strided == data_strided_copy)

    data_strided_copy = copyRowVectorFromConstRef(data_strided)
    assert np.all(data_strided == data_strided_copy)


def test_create_ref(mat):
    print("[asRef(mat)]")
    ref = asRef(mat)
    assert np.array_equal(ref, mat), f"ref=\n{ref}\nmat=\n{mat}"
    assert not (ref.flags.owndata)
    assert ref.flags.writeable


def test_create_const_ref(mat):
    print("[asConstRef]")
    const_ref = asConstRef(mat)
    assert np.array_equal(const_ref, mat), f"ref=\n{const_ref}\nmat=\n{mat}"
    assert not (const_ref.flags.writeable)
    assert not (const_ref.flags.owndata)


def test_edit_block(rows, cols):
    print("set mat data to arange()")
    mat.fill(0.0)
    mat[:, :] = np.arange(rows * cols).reshape(rows, cols)
    mat0 = mat.copy()
    for i, rowsize, colsize in ([0, 3, 2], [1, 1, 2], [0, 3, 1]):
        print(f"taking block [{i}:{rowsize + i}, {0}:{colsize}]")
        B = getBlock(mat, i, 0, rowsize, colsize)
        B = B.reshape(rowsize, colsize)
        lhs = mat[i : rowsize + i, :colsize]
        assert np.array_equal(lhs, B), f"got lhs\n{lhs}\nrhs B=\n{B}"

        B[:] = 1.0
        rhs = np.ones((rowsize, colsize))
        assert np.array_equal(mat[i : rowsize + i, :colsize], rhs)

        mat[:, :] = mat0

    mat.fill(0.0)
    mat_copy = mat.copy()
    print("[editBlock]")
    editBlock(mat, 0, 0, 3, 2)
    mat_copy[:3, :2] = np.arange(6).reshape(3, 2)

    assert np.array_equal(mat, mat_copy)

    class ModifyBlockImpl(modify_block):
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


def do_test(mat):
    test_fill_print(mat)
    test_create_ref_to_static(mat)
    test_create_const_ref(mat)
    test_create_ref(mat)
    test_edit_block(rows, cols)
    test_read_block()
    print("=" * 10)


if __name__ == "__main__":
    rows = 8
    cols = 10

    mat = np.ones((rows, cols), order="F")
    mat[0, 0] = 0
    mat[1:5, 1:5] = 6
    do_test(mat)

    # mat2 = np.ones((rows, cols))
    # mat2[:2, :5] = 0.
    # mat2[2:4, 1:4] = 2
    # mat2[:, -1] = 3
    # do_test(mat2)
