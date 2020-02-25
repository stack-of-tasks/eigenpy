import numpy as np
from eigen_ref import *

def test(mat):

  printMatrix(mat)
  fill(mat,1.)
  printMatrix(mat)
  assert np.array_equal(mat,np.full(mat.shape,1.))

rows = 10
cols = 30

mat = np.array(np.zeros((rows,cols)))

test(mat)
