from __future__ import print_function

import numpy as np
import matrix as eigenpy

verbose = True

if verbose:
    print("===> From empty MatrixXd to Py", flush=True)
M = eigenpy.emptyMatrix()
assert M.shape == (0, 0)

if verbose:
    print("===> From empty VectorXd to Py", flush=True)
v = eigenpy.emptyVector()
assert v.shape == (0,)

if verbose:
    print("===> From MatrixXd to Py", flush=True)
M = eigenpy.naturals(3, 3, verbose)
Mcheck = np.reshape(np.array(range(9), np.double), [3, 3])
assert np.array_equal(Mcheck, M)

if verbose:
    print("===> From Matrisx3d to Py", flush=True)
M33 = eigenpy.naturals33(verbose)
assert np.array_equal(Mcheck, M33)

if verbose:
    print("===> From VectorXd to Py", flush=True)
v = eigenpy.naturalsX(3, verbose)
vcheck = np.array(range(3), np.double).T
assert np.array_equal(vcheck, v)

if verbose:
    print("===> From Py to Eigen::MatrixXd", flush=True)
Mref = np.reshape(np.array(range(64), np.double), [8, 8])

# Test base function
Mref_from_base = eigenpy.base(Mref)
assert np.array_equal(Mref, Mref_from_base)

# Test plain function
Mref_from_plain = eigenpy.plain(Mref)
assert np.array_equal(Mref, Mref_from_plain)

if verbose:
    print("===> Matrix 8x8", flush=True)
M = Mref
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 0:3x0:3", flush=True)
M = Mref[0:3, 0:3]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 1:3x1:3", flush=True)
M = Mref[1:3, 1:3]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 1:5:2x1:5:2", flush=True)
M = Mref[1:5:2, 1:5:2]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 1:8:3x1:5", flush=True)
M = Mref[1:8:3, 1:5]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block transpose 1:8:3x1:6:2")
M = Mref[1:8:3, 0:6:2].T
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 1x0:6:2", flush=True)
M = Mref[1:2, 0:6:2]
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 1x0:6:2 tanspose", flush=True)
M = Mref[1:2, 0:6:2].T
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1", flush=True)
M = Mref[0:6:2, 1:2]
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1 tanspose", flush=True)
M = Mref[0:6:2, 1:2].T
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1 1 dim", flush=True)
M = Mref[0:6:2, 1].T
# TODO
# assert( np.array_equal(M.T,eigenpy.reflexV(M,verbose)) );

if verbose:
    print("===> Block Vector 0:6:2x1", flush=True)
M = Mref[0:6:2, 1:2]
assert np.array_equal(M.squeeze(), eigenpy.reflexV(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1 transpose", flush=True)
M = Mref[0:6:2, 1:2].T
# TODO
# assert( np.array_equal(M.T,eigenpy.reflexV(M,verbose)) );

if verbose:
    print("===> Block Vector 0:3x0:6:2 ")
M = Mref[0:3, 0:6:2]
assert np.array_equal(M, eigenpy.reflex33(M, verbose))

if verbose:
    print("===> Block Vector 0:3x0:6:2 T")
M = Mref[0:3, 0:6].T
# TODO
# try:
# assert( np.array_equal(M,eigenpy.reflex33(M,verbose)) );
# except eigenpy.Exception as e:
# if verbose: print("As expected, got the following /ROW/ error:", e.message)

# TODO
# M = Mref[0:3,1:2]
# assert( np.array_equal(M,eigenpy.reflex3(M,verbose)) );

value = 2.0
mat1x1 = eigenpy.matrix1x1(value)
assert mat1x1.size == 1
assert mat1x1[0, 0] == value

vec1x1 = eigenpy.vector1x1(value)
assert vec1x1.size == 1
assert vec1x1[0] == value

# test registration of matrix6
mat6 = eigenpy.matrix6(0.0)
assert mat6.size == 36

# test RowMajor

mat = np.arange(0, 10).reshape(2, 5)
assert (eigenpy.asRowMajorFromColMajorMatrix(mat) == mat).all()
assert (eigenpy.asRowMajorFromRowMajorMatrix(mat) == mat).all()

vec = np.arange(0, 10)
assert (eigenpy.asRowMajorFromColMajorMatrix(vec) == vec).all()
assert (eigenpy.asRowMajorFromColMajorVector(vec) == vec).all()
assert (eigenpy.asRowMajorFromRowMajorMatrix(vec) == vec).all()
assert (eigenpy.asRowMajorFromRowMajorVector(vec) == vec).all()
