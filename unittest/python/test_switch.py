from __future__ import print_function

import eigenpy
import numpy as np

quat = eigenpy.Quaternion()
# By default, we convert as numpy.matrix
coeffs_vector = quat.coeffs() 
print(type(coeffs_vector))

assert isinstance(coeffs_vector,np.matrixlib.defmatrix.matrix)

# Switch to numpy.array
eigenpy.switchToNumpyArray()
coeffs_array = quat.coeffs()
print(type(coeffs_array))

assert isinstance(coeffs_vector,np.ndarray)
