import time  # noqa
import timeit  # noqa

import numpy as np
from IPython import get_ipython

import eigenpy

ipython = get_ipython()

quat = eigenpy.Quaternion()
a = [0.0, 0.0, 0.0]

cmd1 = "timeit np.array(a)"
print("\n")
print(cmd1)
ipython.magic(cmd1)
print("\n")

cmd2 = "timeit np.matrix(a)"
print(cmd2)
ipython.magic(cmd2)
print("\n")

cmd4 = "timeit quat.coeffs()"
print(cmd4)
ipython.magic(cmd4)
print("\n")

cmd5 = "timeit np.asmatrix(quat.coeffs())"
print(cmd5)
ipython.magic(cmd5)
print("\n")

a_matrix = np.matrix(a)
cmd6 = "timeit np.asarray(a_matrix)"
print(cmd6)
ipython.magic(cmd6)
print("\n")
