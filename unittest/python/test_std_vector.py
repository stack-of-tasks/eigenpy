import numpy as np
import eigenpy
import inspect
import vector
from vector import printVectorOfMatrix, printVectorOf3x3, copyStdVector

np.random.seed(0)

l1 = [np.random.randn(3), np.random.randn(2)]
l2 = eigenpy.StdVec_VectorXd(l1)
l3 = [np.random.randn(2, 2), np.random.randn(3, 1), np.random.randn(4, 2)]


def checkAllValues(li1, li2):
    assert len(li1) == len(li2)
    n = len(li1)
    for i in range(n):
        assert np.array_equal(li1[i], li2[i])


checkAllValues(l1, l2)
checkAllValues(l1, copyStdVector(l1))


printVectorOfMatrix(l1)
print()
printVectorOfMatrix(l2)
print()
printVectorOfMatrix(l3)
print()


l4 = [np.random.randn(3, 3) for _ in range(4)]
assert "StdVec_Mat3d" in printVectorOf3x3.__doc__
printVectorOf3x3(l4)
print()

l4_copy = copyStdVector(l4)
assert isinstance(l4_copy, eigenpy.StdVec_MatrixXd)

l4_copy2 = vector.copyStdVec_3x3(l4)
assert isinstance(l4_copy2, vector.StdVec_Mat3d)


def checkZero(l):
    for x in l:
        assert np.allclose(x, 0.0), "x = {}".format(x)


print("l1:")
vector.setZero(l1)
print(l1)
checkZero(l1)

print("l2:")
l2_py = l2.tolist()
vector.setZero(l2_py)
print(l2_py)
checkZero(l2_py)

print("l3:")
vector.setZero(l3)
checkZero(l3)
