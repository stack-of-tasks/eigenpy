import pprint

import numpy as np
import std_vector
from std_vector import copyStdVector, printVectorOf3x3, printVectorOfMatrix

import eigenpy

rng = np.random.default_rng(0)

l1 = [rng.standard_normal(3), rng.standard_normal(2)]
l2 = eigenpy.StdVec_VectorXd(l1)
l3 = [rng.standard_normal((2, 2)), rng.standard_normal((3, 1))]
l3.append(np.asfortranarray(np.eye(2)))
l3.append(np.eye(2))
l4 = [rng.standard_normal((3, 3)).T for _ in range(3)]
l4[-1] = l4[-1].T
l5 = [rng.standard_normal((2, 2)).T for _ in range(3)]


def checkAllValues(li1, li2):
    assert len(li1) == len(li2)
    n = len(li1)
    for i in range(n):
        assert np.array_equal(li1[i], li2[i])


checkAllValues(l1, l2)
checkAllValues(l1, copyStdVector(l1))

l2[0][:2] = 0.0
assert np.allclose(l2[0][:2], 0.0)

print("l1")
printVectorOfMatrix(l1)
print("l2")
printVectorOfMatrix(l2)
print("l3")
printVectorOfMatrix(l3)


l4_copy = copyStdVector(l4)
assert isinstance(l4_copy, eigenpy.StdVec_MatrixXd)

assert "StdVec_Mat3d" in printVectorOf3x3.__doc__
printVectorOf3x3(l4)

l4_copy2 = std_vector.copyStdVec_3x3(l4)
assert isinstance(l4_copy2, std_vector.StdVec_Mat3d)


def checkZero(v):
    for x in v:
        assert np.allclose(x, 0.0), f"x = {x}"


print("Check setZero() works:")
std_vector.setZero(l1)
print("l1:")
print(l1)
checkZero(l1)
print("-----------------")

print("l2:")
l2_py = l2.tolist()
std_vector.setZero(l2_py)
pprint.pprint(l2_py)
checkZero(l2_py)
print("-----------------")

l3_copy = copyStdVector(l3)
print("l3_std:")
std_vector.setZero(l3_copy)
pprint.pprint(list(l3_copy))
checkZero(l3_copy)
print("-----------------")

# print("l3_python:")
# vector.setZero(l3)
# pprint.pprint(list(l3))
# checkZero(l3)
# print("-----------------")

# print("l4:")
# vector.setZero(l4)
# pprint.pprint(list(l4))
# checkZero(l4)

# Test StdVec_Mat2d that had been registered
# before calling exposeStdVectorEigenSpecificType

# Test conversion and tolistl5 == l5_copy == l5_py
l5_copy = std_vector.StdVec_Mat2d(l5)
l5_py = l5_copy.tolist()
checkAllValues(l5, l5_copy)
checkAllValues(l5, l5_py)

# Test mutable __getitem__
l5[0][:] = 0.0
assert np.allclose(l5[0], 0.0)
