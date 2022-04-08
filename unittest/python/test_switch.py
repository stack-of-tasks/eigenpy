from __future__ import print_function

import eigenpy
import numpy as np

eigenpy.switchToNumpyMatrix()
quat = eigenpy.Quaternion()
# By default, we convert as numpy.matrix
coeffs_vector = quat.coeffs()
print(type(coeffs_vector))

assert isinstance(coeffs_vector, np.matrixlib.defmatrix.matrix)
assert eigenpy.getNumpyType() == np.matrix

# Switch to numpy.array
eigenpy.switchToNumpyArray()
coeffs_array = quat.coeffs()
print(type(coeffs_array))

assert isinstance(coeffs_vector, np.ndarray)
assert eigenpy.getNumpyType() == np.ndarray
