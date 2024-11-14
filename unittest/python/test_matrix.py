import platform

import matrix as eigenpy
import numpy as np

verbose = True

if verbose:
    print("===> From empty MatrixXd to Py")
M = eigenpy.emptyMatrix()
assert M.shape == (0, 0)

if verbose:
    print("===> From empty VectorXd to Py")
v = eigenpy.emptyVector()
assert v.shape == (0,)

if verbose:
    print("===> From Py to Matrix1")
eigenpy.matrix1x1(np.array([1]))

if verbose:
    print("===> From MatrixXd to Py")
M = eigenpy.naturals(3, 3, verbose)
Mcheck = np.reshape(np.array(range(9), np.double), [3, 3])
assert np.array_equal(Mcheck, M)

if verbose:
    print("===> From Matrix3d to Py")
M33 = eigenpy.naturals33(verbose)
assert np.array_equal(Mcheck, M33)

if verbose:
    print("===> From VectorXd to Py")
v = eigenpy.naturalsX(3, verbose)
vcheck = np.array(range(3), np.double).T
assert np.array_equal(vcheck, v)

if verbose:
    print("===> From Py to Eigen::MatrixXd")
if verbose:
    print("===> From Py to Eigen::MatrixXd")
if verbose:
    print("===> From Py to Eigen::MatrixXd")
Mref = np.reshape(np.array(range(64), np.double), [8, 8])

# Test base function
Mref_from_base = eigenpy.base(Mref)
assert np.array_equal(Mref, Mref_from_base)

# Test plain function
Mref_from_plain = eigenpy.plain(Mref)
assert np.array_equal(Mref, Mref_from_plain)

if verbose:
    print("===> Matrix 8x8")
M = Mref
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 0:3x0:3")
M = Mref[0:3, 0:3]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 1:3x1:3")
M = Mref[1:3, 1:3]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 1:5:2x1:5:2")
M = Mref[1:5:2, 1:5:2]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block 1:8:3x1:5")
M = Mref[1:8:3, 1:5]
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block transpose 1:8:3x1:6:2")
M = Mref[1:8:3, 0:6:2].T
assert np.array_equal(M, eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 1x0:6:2")
M = Mref[1:2, 0:6:2]
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 1x0:6:2 tanspose")
M = Mref[1:2, 0:6:2].T
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1")
M = Mref[0:6:2, 1:2]
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1 tanspose")
M = Mref[0:6:2, 1:2].T
assert np.array_equal(M.squeeze(), eigenpy.reflex(M, verbose))

if verbose:
    print("===> From Py to Eigen::VectorXd")
if verbose:
    print("===> From Py to Eigen::VectorXd")
if verbose:
    print("===> From Py to Eigen::VectorXd")

if verbose:
    print("===> Block Vector 0:6:2x1 1 dim")
M = Mref[0:6:2, 1].T
# TODO
# assert( np.array_equal(M.T,eigenpy.reflexV(M,verbose)) );

if verbose:
    print("===> Block Vector 0:6:2x1")
M = Mref[0:6:2, 1:2]
assert np.array_equal(M.squeeze(), eigenpy.reflexV(M, verbose))

if verbose:
    print("===> Block Vector 0:6:2x1 transpose")
M = Mref[0:6:2, 1:2].T
# TODO
# assert( np.array_equal(M.T,eigenpy.reflexV(M,verbose)) );

if verbose:
    print("===> From Py to Eigen::Matrix3d")
if verbose:
    print("===> From Py to Eigen::Matrix3d")
if verbose:
    print("===> From Py to Eigen::Matrix3d")

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

if verbose:
    print("===> From Py to Eigen::Vector3d")
if verbose:
    print("===> From Py to Eigen::Vector3d")
if verbose:
    print("===> From Py to Eigen::Vector3d")

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

# Test numpy -> Eigen -> numpy for all same type


def test_conversion(function, dtype):
    input_array = np.array([1, 0], dtype=dtype)
    assert input_array.dtype == dtype
    output_array = function(input_array)
    assert output_array.dtype == dtype
    assert (output_array == input_array).all()


bool_t = np.dtype(np.bool_)
int8_t = np.dtype(np.int8)
uint8_t = np.dtype(np.uint8)
int16_t = np.dtype(np.int16)
uint16_t = np.dtype(np.uint16)
int32_t = np.dtype(np.int32)
uint32_t = np.dtype(np.uint32)
int64_t = np.dtype(np.int64)
uint64_t = np.dtype(np.uint64)
# On Windows long is a 32 bits integer but is a different type than int32.
long_t = np.dtype(np.int32 if platform.system() == "Windows" else np.int64)
ulong_t = np.dtype(np.uint32 if platform.system() == "Windows" else np.uint64)
longlong_t = np.dtype(np.longlong)
ulonglong_t = np.dtype(np.ulonglong)

float32_t = np.dtype(np.float32)
float64_t = np.dtype(np.float64)

complex64_t = np.dtype(np.complex64)
complex128_t = np.dtype(np.complex128)
complex256_t = np.dtype(np.clongdouble)


test_conversion(eigenpy.copyBoolToBool, bool_t)

test_conversion(eigenpy.copyInt8ToInt8, int8_t)
test_conversion(eigenpy.copyCharToChar, int8_t)
test_conversion(eigenpy.copyUCharToUChar, uint8_t)

test_conversion(eigenpy.copyInt16ToInt16, int16_t)
test_conversion(eigenpy.copyUInt16ToUInt16, uint16_t)

test_conversion(eigenpy.copyInt32ToInt32, int32_t)
test_conversion(eigenpy.copyUInt32ToUInt32, uint32_t)

test_conversion(eigenpy.copyInt64ToInt64, int64_t)
test_conversion(eigenpy.copyUInt64ToUInt64, uint64_t)

test_conversion(eigenpy.copyLongToLong, long_t)
test_conversion(eigenpy.copyULongToULong, ulong_t)

# On Windows long long is an int64_t alias.
# The numpy dtype match between longlong and int64.

# On Linux long long is a 64 bits integer but is a different type than int64_t.
# The numpy dtype doesn't match and it's not an issue since C++ type are different.

# On Mac long long is an int64_t and long alias.
# This is an issue because longlong numpy dtype is different than long numpy dtype
# but long and long long are the same type in C++.
# The test should pass thanks to the promotion code.
test_conversion(eigenpy.copyLongLongToLongLong, longlong_t)
test_conversion(eigenpy.copyULongLongToULongLong, ulonglong_t)

test_conversion(eigenpy.copyFloatToFloat, float32_t)
test_conversion(eigenpy.copyDoubleToDouble, float64_t)
# On Windows and Mac longdouble is 64 bits
test_conversion(eigenpy.copyLongDoubleToLongDouble, np.dtype(np.longdouble))

test_conversion(eigenpy.copyCFloatToCFloat, complex64_t)
test_conversion(eigenpy.copyCDoubleToCDouble, complex128_t)
# On Windows and Mac clongdouble is 128 bits
test_conversion(eigenpy.copyCLongDoubleToCLongDouble, complex256_t)


# Test numpy -> Eigen -> numpy promotion


def test_conversion_promotion(function, input_dtype, output_dtype):
    input_array = np.array([1, 0], dtype=input_dtype)
    assert input_array.dtype == input_dtype
    output_array = function(input_array)
    assert output_array.dtype == output_dtype
    assert (output_array == input_array).all()


# Test bool to other type
test_conversion_promotion(eigenpy.copyInt8ToInt8, bool_t, int8_t)
test_conversion_promotion(eigenpy.copyUCharToUChar, bool_t, uint8_t)
test_conversion_promotion(eigenpy.copyInt16ToInt16, bool_t, int16_t)
test_conversion_promotion(eigenpy.copyUInt16ToUInt16, bool_t, uint16_t)
test_conversion_promotion(eigenpy.copyInt32ToInt32, bool_t, int32_t)
test_conversion_promotion(eigenpy.copyUInt32ToUInt32, bool_t, uint32_t)
test_conversion_promotion(eigenpy.copyInt64ToInt64, bool_t, int64_t)
test_conversion_promotion(eigenpy.copyUInt64ToUInt64, bool_t, uint64_t)
test_conversion_promotion(eigenpy.copyLongToLong, bool_t, long_t)
test_conversion_promotion(eigenpy.copyULongToULong, bool_t, ulong_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, bool_t, int64_t)
test_conversion_promotion(eigenpy.copyULongLongToULongLong, bool_t, uint64_t)

# Test int8 to other type
test_conversion_promotion(eigenpy.copyInt16ToInt16, int8_t, int16_t)
test_conversion_promotion(eigenpy.copyInt16ToInt16, uint8_t, int16_t)
test_conversion_promotion(eigenpy.copyUInt16ToUInt16, uint8_t, uint16_t)
test_conversion_promotion(eigenpy.copyInt32ToInt32, int8_t, int32_t)
test_conversion_promotion(eigenpy.copyInt32ToInt32, uint8_t, int32_t)
test_conversion_promotion(eigenpy.copyUInt32ToUInt32, uint8_t, uint32_t)
test_conversion_promotion(eigenpy.copyInt64ToInt64, int8_t, int64_t)
test_conversion_promotion(eigenpy.copyInt64ToInt64, uint8_t, int64_t)
test_conversion_promotion(eigenpy.copyUInt64ToUInt64, uint8_t, uint64_t)
test_conversion_promotion(eigenpy.copyLongToLong, int8_t, long_t)
test_conversion_promotion(eigenpy.copyLongToLong, uint8_t, long_t)
test_conversion_promotion(eigenpy.copyULongToULong, uint8_t, ulong_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, int8_t, int64_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, uint8_t, int64_t)
test_conversion_promotion(eigenpy.copyULongLongToULongLong, uint8_t, uint64_t)

# Test int16 to other type
test_conversion_promotion(eigenpy.copyInt32ToInt32, int16_t, int32_t)
test_conversion_promotion(eigenpy.copyInt32ToInt32, uint16_t, int32_t)
test_conversion_promotion(eigenpy.copyUInt32ToUInt32, uint16_t, uint32_t)
test_conversion_promotion(eigenpy.copyInt64ToInt64, int16_t, int64_t)
test_conversion_promotion(eigenpy.copyInt64ToInt64, uint16_t, int64_t)
test_conversion_promotion(eigenpy.copyUInt64ToUInt64, uint16_t, uint64_t)
test_conversion_promotion(eigenpy.copyLongToLong, int16_t, long_t)
test_conversion_promotion(eigenpy.copyLongToLong, uint16_t, long_t)
test_conversion_promotion(eigenpy.copyULongToULong, uint16_t, ulong_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, int16_t, int64_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, uint16_t, int64_t)
test_conversion_promotion(eigenpy.copyULongLongToULongLong, uint16_t, uint64_t)

# Test int32 to other type
test_conversion_promotion(eigenpy.copyInt64ToInt64, int32_t, int64_t)
test_conversion_promotion(eigenpy.copyInt64ToInt64, uint32_t, int64_t)
test_conversion_promotion(eigenpy.copyUInt64ToUInt64, uint32_t, uint64_t)
test_conversion_promotion(eigenpy.copyLongToLong, int32_t, long_t)
test_conversion_promotion(eigenpy.copyLongToLong, uint32_t, long_t)
test_conversion_promotion(eigenpy.copyULongToULong, uint32_t, ulong_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, int32_t, int64_t)
test_conversion_promotion(eigenpy.copyLongLongToLongLong, uint32_t, int64_t)
test_conversion_promotion(eigenpy.copyULongLongToULongLong, uint32_t, uint64_t)

# Test float to double
test_conversion_promotion(eigenpy.copyDoubleToDouble, float32_t, float64_t)

# Test complex to other type
test_conversion_promotion(eigenpy.copyCDoubleToCDouble, complex64_t, complex128_t)

test_conversion_promotion(
    eigenpy.copyCLongDoubleToCLongDouble,
    complex64_t,
    complex256_t,
)

# Only Linux store complex double into 258 bits.
# Then, 128 bits complex can be promoted.
if platform.system() == "Linux":
    test_conversion_promotion(
        eigenpy.copyCLongDoubleToCLongDouble,
        complex128_t,
        complex256_t,
    )
