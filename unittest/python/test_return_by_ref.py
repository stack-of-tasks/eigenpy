import numpy as np
import return_by_ref
from return_by_ref import Matrix, RowMatrix, Vector


def test_shared(mat):
    m_ref = mat.ref()
    m_ref.fill(0)
    m_copy = mat.copy()
    assert np.array_equal(m_ref, m_copy)

    m_const_ref = mat.const_ref()
    assert np.array_equal(m_const_ref, m_copy)
    assert np.array_equal(m_const_ref, m_ref)

    m_ref.fill(1)
    assert not np.array_equal(m_ref, m_copy)
    assert np.array_equal(m_const_ref, m_ref)

    try:
        m_const_ref.fill(2)
        assert False
    except Exception:
        assert True


def test_not_shared(mat):
    m_ref = mat.ref()
    m_ref.fill(100.0)
    m_copy = mat.copy()
    assert not np.array_equal(m_ref, m_copy)

    m_const_ref = mat.const_ref()
    assert np.array_equal(m_const_ref, m_copy)
    assert not np.array_equal(m_const_ref, m_ref)

    m_ref.fill(10.0)
    assert not np.array_equal(m_ref, m_copy)
    assert not np.array_equal(m_const_ref, m_ref)

    try:
        m_const_ref.fill(2)
        assert True
    except Exception:
        assert False


rows = 10
cols = 30

mat = Matrix(rows, cols)
row_mat = RowMatrix(rows, cols)
vec = Vector(rows, 1)

test_shared(mat)
test_shared(row_mat)
test_shared(vec)

return_by_ref.sharedMemory(False)
test_not_shared(mat)
test_not_shared(row_mat)
test_not_shared(vec)
