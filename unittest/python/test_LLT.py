import eigenpy
eigenpy.switchToNumpyArray()

import numpy as np
import numpy.linalg as la

dim = 100
A = np.random.rand(dim,dim)

A = (A + A.T)*0.5 + np.diag(10. + np.random.rand(dim))

llt = eigenpy.LLT(A)

L = llt.matrixL() 

assert eigenpy.is_approx(L.dot(np.transpose(L)),A)
