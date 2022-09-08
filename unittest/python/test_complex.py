from __future__ import print_function

import numpy as np
from complex import switchToNumpyArray, real, imag, ascomplex

switchToNumpyArray()

rows = 10
cols = 20


def test(dtype):
    Z = np.zeros((rows, cols), dtype=dtype)
    Z.real = np.random.rand(rows, cols)
    Z.imag = np.random.rand(rows, cols)

    Z_real = real(Z)
    assert (Z_real == Z.real).all()
    Z_imag = imag(Z)
    assert (Z_imag == Z.imag).all()

    Y = np.ones((rows, cols))
    Y_complex = ascomplex(Y)
    assert (Y_complex.real == Y).all()
    assert (Y_complex.imag == np.zeros((rows, cols))).all()


# Float
test(np.csingle)
# Double
test(np.cdouble)
# Long Double
test(np.clongdouble)
