import numpy as np
import eigenpy
import inspect
import vector
from vector import printVectorOfMatrix, printVectorOf3x3, copyStdVector

np.random.seed(0)

l1 = [np.random.randn(3), np.random.randn(2)]
l2 = eigenpy.StdVec_VectorXd(l1)
l3 = [np.random.randn(2, 2), np.random.randn(1, 2), np.random.randn(2, 6)]


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

l4_copy = copyStdVector(l4)
assert isinstance(l4_copy, eigenpy.StdVec_MatrixXd)
print(l4_copy)

l4_copy2 = vector.copyStdVec_3x3(l4)
assert isinstance(l4_copy2, vector.StdVec_Mat3d)
print(l4_copy2)
