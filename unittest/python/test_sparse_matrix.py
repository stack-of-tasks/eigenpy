from __future__ import print_function

import numpy as np
import sparse_matrix

m = sparse_matrix.emptyMatrix()
assert m.shape == (0, 0)

v = sparse_matrix.emptyVector()
assert v.shape == (0, 0)

m = sparse_matrix.matrix1x1(2)
assert m.toarray() == np.array([2])

v = sparse_matrix.vector1x1(2)
assert v.toarray() == np.array([2])

size = 100
diag_values = np.random.rand(100)
diag_mat = sparse_matrix.diagonal(diag_values)
assert (diag_mat.toarray() == np.diag(diag_values)).all()

diag_mat_copy = sparse_matrix.copy(diag_mat)
assert (diag_mat_copy != diag_mat).nnz == 0
