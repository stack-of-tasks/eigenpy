import eigenpy

quat = eigenpy.Quaternion()

# Switch to numpy.array
coeffs_vector = quat.coeffs()
assert len(coeffs_vector.shape) == 1
