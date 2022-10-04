import numpy as np
import eigenpy

x0 = np.random.randn(3)
l1 = [x0, x0, x0]
l2 = eigenpy.StdVec_VectorXd(3, x0)


def checkAllValues(li1, li2):
    assert len(li1) == len(li2)
    n = len(li1)
    for i in range(n):
        assert np.array_equal(li1[i], li2[i])


checkAllValues(l1, l2)
