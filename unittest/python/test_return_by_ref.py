from return_by_ref import Matrix, RowMatrix, Vector
import numpy as np

def test(mat):

  m_ref = mat.ref()
  m_ref.fill(0)
  m_copy = mat.copy()
  assert np.array_equal(m_ref,m_copy)

  m_const_ref = mat.const_ref()
  assert np.array_equal(m_const_ref,m_copy)
  assert np.array_equal(m_const_ref,m_ref)

  m_ref.fill(1)
  assert not np.array_equal(m_ref,m_copy)
  assert np.array_equal(m_const_ref,m_ref)

  try:
    m_const_ref.fill(2)
    assert False
  except:
    assert True

rows = 10
cols = 30

mat = Matrix(rows,cols)
row_mat = RowMatrix(rows,cols)
vec = Vector(rows,1)

test(mat)
test(row_mat)
test(vec)
