import numpy as np
import sparse_matrix
from scipy.sparse import csr_matrix

m = sparse_matrix.emptyMatrix()
assert m.shape == (0, 0)

v = sparse_matrix.emptyVector()
assert v.shape == (0, 0)

m = sparse_matrix.matrix1x1(2)
assert m.toarray() == np.array([2])

v = sparse_matrix.vector1x1(2)
assert v.toarray() == np.array([2])

rng = np.random.default_rng()
diag_values = rng.random(10)
diag_mat = sparse_matrix.diagonal(diag_values)
assert (diag_mat.toarray() == np.diag(diag_values)).all()

diag_mat_copy = sparse_matrix.copy(diag_mat)
assert (diag_mat_copy != diag_mat).nnz == 0

diag_mat_csr = csr_matrix(diag_mat)
assert (sparse_matrix.copy(diag_mat_csr) != diag_mat_csr).nnz == 0

# test zero matrix
zero_mat = csr_matrix(np.zeros((10, 1)))
zero_mat_copy = sparse_matrix.copy(zero_mat)
assert (zero_mat_copy != zero_mat).nnz == 0
