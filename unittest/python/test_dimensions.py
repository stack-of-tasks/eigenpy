from __future__ import print_function

import eigenpy
import numpy as np

quat = eigenpy.Quaternion()
# By default, we convert as numpy.matrix
eigenpy.switchToNumpyMatrix()
coeffs_vector = quat.coeffs()
assert len(coeffs_vector.shape) == 2

# Switch to numpy.array
eigenpy.switchToNumpyArray()
coeffs_vector = quat.coeffs()
assert len(coeffs_vector.shape) == 1
