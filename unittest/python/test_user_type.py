import user_type

rows = 10
cols = 20

def test(mat):
  mat.fill(mat.dtype.type(10.))
  mat_copy = mat.copy()
  assert (mat == mat_copy).all()
  assert not (mat != mat_copy).all()

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
