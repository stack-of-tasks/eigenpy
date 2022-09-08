import user_type
import numpy as np

# from packaging import version

rows = 10
cols = 20


def test(dtype):
    mat = np.array(np.ones((rows, cols)).astype(np.intc), dtype=dtype)
    mat = np.random.rand(rows, cols).astype(dtype)
    mat_copy = mat.copy()
    assert (mat == mat_copy).all()
    assert not (mat != mat_copy).all()

    # if version.parse(np.__version__) >= version.parse("1.21.0"):
    # # check if it fixes for new version of NumPy
    # mat.fill(mat.dtype.type(20.0))
    # mat_copy = mat.copy()
    # assert (mat == mat_copy).all()
    # assert not (mat != mat_copy).all()

    mat_op = mat + mat
    mat_op = mat.copy(order="F") + mat.copy(order="C")

    mat_op = mat - mat
    mat_op = mat * mat
    mat_op = mat.dot(mat.T)
    mat_op = mat / mat

    mat_op = -mat  # noqa

    assert (mat >= mat).all()
    assert (mat <= mat).all()
    assert not (mat > mat).all()
    assert not (mat < mat).all()

    mat2 = mat.dot(mat.T)
    mat2_ref = mat.astype(np.double).dot(mat.T.astype(np.double))
    assert np.isclose(mat2.astype(np.double), mat2_ref).all()
    if np.__version__ >= "1.17.0":
        mat2 = np.matmul(mat, mat.T)
        assert np.isclose(mat2.astype(np.double), mat2_ref).all()


def test_cast(from_dtype, to_dtype):
    np.can_cast(from_dtype, to_dtype)

    from_mat = np.zeros((rows, cols), dtype=from_dtype)
    to_mat = from_mat.astype(dtype=to_dtype)  # noqa


test(user_type.CustomDouble)

test_cast(user_type.CustomDouble, np.double)
test_cast(np.double, user_type.CustomDouble)

test_cast(user_type.CustomDouble, np.int64)
test_cast(np.int64, user_type.CustomDouble)

test_cast(user_type.CustomDouble, np.int32)
test_cast(np.int32, user_type.CustomDouble)

test(user_type.CustomFloat)

v = user_type.CustomDouble(1)
a = np.array(v)
assert type(v) == a.dtype.type
