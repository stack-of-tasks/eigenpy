from __future__ import print_function

import eigenpy
import numpy as np

import time
import timeit

from IPython import get_ipython

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

eigenpy.switchToNumpyMatrix()
print("----------------------")
print("switch to numpy matrix")
print("----------------------")
print("\n")

cmd3 = "timeit quat.coeffs()"
print(cmd3)
ipython.magic(cmd3)
print("\n")

eigenpy.switchToNumpyArray()
print("---------------------")
print("switch to numpy array")
print("---------------------")
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
