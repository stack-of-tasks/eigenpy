import eigenpy
eigenpy.switchToNumpyArray()

import numpy as np
import numpy.linalg as la

dim = 100
A = np.random.rand(dim,dim)

A = (A + A.T)*0.5 + np.diag(10. + np.random.rand(dim))

ldlt = eigenpy.LDLT(A)

L = ldlt.matrixL() 
D = ldlt.vectorD() 
P = ldlt.transpositionsP() 

assert eigenpy.is_approx(np.transpose(P).dot(L.dot(np.diag(D).dot(np.transpose(L).dot(P)))),A)
