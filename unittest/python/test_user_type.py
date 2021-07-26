import user_type
import numpy as np
#from packaging import version

rows = 10
cols = 20

def test(mat):
  mat[:] = mat.dtype.type(10.)
  mat_copy = mat.copy()
  assert (mat == mat_copy).all()
  assert not (mat != mat_copy).all()

#  if version.parse(np.__version__) >= version.parse("1.21.0"): # check if it fixes for new versio of NumPy 
#    mat.fill(mat.dtype.type(20.))
#    mat_copy = mat.copy()
#    assert((mat == mat_copy).all())
#    assert(not (mat != mat_copy).all())

  mat_op = mat + mat
  mat_op = mat.copy(order='F') + mat.copy(order='C')
  
  mat_op = mat - mat
  mat_op = mat * mat
  mat_op = mat.dot(mat.T)
  mat_op = mat / mat

  mat_op = -mat;

  assert (mat >= mat).all()
  assert (mat <= mat).all()
  assert not (mat > mat).all()
  assert not (mat < mat).all()

mat = user_type.create_double(rows,cols)
test(mat)

mat = user_type.create_float(rows,cols)
test(mat)

v = user_type.CustomDouble(1)
a = np.array(v)
assert type(v) == a.dtype.type
