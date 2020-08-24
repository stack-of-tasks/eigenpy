import numpy as np
from eigen_ref import *

def test(mat):

  printMatrix(mat)
  fill(mat,1.)
  printMatrix(mat)
  assert np.array_equal(mat,np.full(mat.shape,1.))

  A_ref = asRef(mat.shape[0],mat.shape[1])
  A_ref.fill(1.)
  A_ref2 = asRef(mat.shape[0],mat.shape[1])

  assert np.array_equal(A_ref,A_ref2)

  A_ref2.fill(0)
  assert np.array_equal(A_ref,A_ref2)
  

rows = 10
cols = 30

mat = np.array(np.zeros((rows,cols)))

test(mat)
