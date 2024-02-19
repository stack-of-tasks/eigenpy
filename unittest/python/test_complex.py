import numpy as np
from complex import ascomplex, imag, real

rows = 10
cols = 20
rng = np.random.default_rng()


def test(dtype):
    Z = np.zeros((rows, cols), dtype=dtype)
    Z.real = rng.random((rows, cols))
    Z.imag = rng.random((rows, cols))

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
