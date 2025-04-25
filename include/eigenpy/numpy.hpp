/*
 * Copyright 2020-2024 INRIA
 */

#ifndef __eigenpy_numpy_hpp__
#define __eigenpy_numpy_hpp__

#include "eigenpy/config.hpp"

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL EIGENPY_ARRAY_API
#endif

// For compatibility with Numpy 2.x. See:
// https://numpy.org/devdocs/reference/c-api/array.html#c.NPY_API_SYMBOL_ATTRIBUTE
#define NPY_API_SYMBOL_ATTRIBUTE EIGENPY_DLLAPI

// When building with MSVC, Python headers use some pragma operator to link
// against the Python DLL.
// Unfortunately, it can link against the wrong build type of the library
// leading to some linking issue.
// Boost::Python provides a helper specifically dedicated to selecting the right
// Python library depending on build type, so let's make use of it.
// Numpy headers drags Python with them. As a result, it
// is necessary to include this helper before including Numpy.
// See: https://github.com/stack-of-tasks/eigenpy/pull/514
#include <boost/python/detail/wrap_python.hpp>

#include <numpy/numpyconfig.h>
#ifdef NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// Allow compiling against NumPy 1.x and 2.x. See:
// https://github.com/numpy/numpy/blob/afea8fd66f6bdbde855f5aff0b4e73eb0213c646/doc/source/reference/c-api/array.rst#L1224
#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#if NPY_ABI_VERSION < 0x02000000
static inline PyArray_ArrFuncs* PyDataType_GetArrFuncs(PyArray_Descr* descr) {
  return descr->f;
}
#endif

/* PEP 674 disallow using macros as l-values
   see : https://peps.python.org/pep-0674/
*/
#if PY_VERSION_HEX < 0x030900A4 && !defined(Py_SET_TYPE)
static inline void _Py_SET_TYPE(PyObject* o, PyTypeObject* type) {
  Py_TYPE(o) = type;
}
#define Py_SET_TYPE(o, type) _Py_SET_TYPE((PyObject*)(o), type)
#endif

#if defined _WIN32 || defined __CYGWIN__
#define EIGENPY_GET_PY_ARRAY_TYPE(array) \
  call_PyArray_MinScalarType(array)->type_num
#else
#define EIGENPY_GET_PY_ARRAY_TYPE(array) PyArray_MinScalarType(array)->type_num
#endif

#include <complex>

namespace eigenpy {
void EIGENPY_DLLAPI import_numpy();
int EIGENPY_DLLAPI PyArray_TypeNum(PyTypeObject* type);

// By default, the Scalar is considered as a Python object
template <typename Scalar, typename Enable = void>
struct NumpyEquivalentType {
  enum { type_code = NPY_USERDEF };
};

template <>
struct NumpyEquivalentType<bool> {
  enum { type_code = NPY_BOOL };
};

template <>
struct NumpyEquivalentType<char> {
  enum { type_code = NPY_INT8 };
};
template <>
struct NumpyEquivalentType<unsigned char> {
  enum { type_code = NPY_UINT8 };
};
template <>
struct NumpyEquivalentType<int8_t> {
  enum { type_code = NPY_INT8 };
};

template <>
struct NumpyEquivalentType<int16_t> {
  enum { type_code = NPY_INT16 };
};
template <>
struct NumpyEquivalentType<uint16_t> {
  enum { type_code = NPY_UINT16 };
};

template <>
struct NumpyEquivalentType<int32_t> {
  enum { type_code = NPY_INT32 };
};
template <>
struct NumpyEquivalentType<uint32_t> {
  enum { type_code = NPY_UINT32 };
};

// On Windows, long is a 32 bytes type but it's a different type than int
// See https://github.com/stack-of-tasks/eigenpy/pull/455
#if defined _WIN32 || defined __CYGWIN__

template <>
struct NumpyEquivalentType<long> {
  enum { type_code = NPY_INT32 };
};
template <>
struct NumpyEquivalentType<unsigned long> {
  enum { type_code = NPY_UINT32 };
};

#endif  // WIN32

template <>
struct NumpyEquivalentType<int64_t> {
  enum { type_code = NPY_INT64 };
};
template <>
struct NumpyEquivalentType<uint64_t> {
  enum { type_code = NPY_UINT64 };
};

// On Mac, long is a 64 bytes type but it's a different type than int64_t
// See https://github.com/stack-of-tasks/eigenpy/pull/455
#if defined __APPLE__

template <>
struct NumpyEquivalentType<long> {
  enum { type_code = NPY_INT64 };
};
template <>
struct NumpyEquivalentType<unsigned long> {
  enum { type_code = NPY_UINT64 };
};

#endif  // MAC

// On Linux, long long is a 64 bytes type but it's a different type than int64_t
// See https://github.com/stack-of-tasks/eigenpy/pull/455
#if defined __linux__

#include <type_traits>

template <typename Scalar>
struct NumpyEquivalentType<
    Scalar,
    typename std::enable_if<!std::is_same<int64_t, long long>::value &&
                            std::is_same<Scalar, long long>::value>::type> {
  enum { type_code = NPY_LONGLONG };
};
template <typename Scalar>
struct NumpyEquivalentType<
    Scalar, typename std::enable_if<
                !std::is_same<uint64_t, unsigned long long>::value &&
                std::is_same<Scalar, unsigned long long>::value>::type> {
  enum { type_code = NPY_ULONGLONG };
};

#endif  // Linux

template <>
struct NumpyEquivalentType<float> {
  enum { type_code = NPY_FLOAT };
};
template <>
struct NumpyEquivalentType<double> {
  enum { type_code = NPY_DOUBLE };
};
template <>
struct NumpyEquivalentType<long double> {
  enum { type_code = NPY_LONGDOUBLE };
};

template <>
struct NumpyEquivalentType<std::complex<float>> {
  enum { type_code = NPY_CFLOAT };
};
template <>
struct NumpyEquivalentType<std::complex<double>> {
  enum { type_code = NPY_CDOUBLE };
};
template <>
struct NumpyEquivalentType<std::complex<long double>> {
  enum { type_code = NPY_CLONGDOUBLE };
};

template <typename Scalar>
bool isNumpyNativeType() {
  if ((int)NumpyEquivalentType<Scalar>::type_code == NPY_USERDEF) return false;
  return true;
}

}  // namespace eigenpy

namespace eigenpy {
#if defined _WIN32 || defined __CYGWIN__
EIGENPY_DLLAPI bool call_PyArray_Check(PyObject*);

EIGENPY_DLLAPI PyObject* call_PyArray_SimpleNew(int nd, npy_intp* shape,
                                                int np_type);

EIGENPY_DLLAPI PyObject* call_PyArray_New(PyTypeObject* py_type_ptr, int nd,
                                          npy_intp* shape, int np_type,
                                          void* data_ptr, int options);

EIGENPY_DLLAPI PyObject* call_PyArray_New(PyTypeObject* py_type_ptr, int nd,
                                          npy_intp* shape, int np_type,
                                          npy_intp* strides, void* data_ptr,
                                          int options);

EIGENPY_DLLAPI int call_PyArray_ObjectType(PyObject*, int);

EIGENPY_DLLAPI PyTypeObject* getPyArrayType();

EIGENPY_DLLAPI PyArray_Descr* call_PyArray_DescrFromType(int typenum);

EIGENPY_DLLAPI void call_PyArray_InitArrFuncs(PyArray_ArrFuncs* funcs);

EIGENPY_DLLAPI int call_PyArray_RegisterDataType(PyArray_DescrProto* dtype);

EIGENPY_DLLAPI int call_PyArray_RegisterCanCast(PyArray_Descr* descr,
                                                int totype,
                                                NPY_SCALARKIND scalar);

EIGENPY_DLLAPI PyArray_Descr* call_PyArray_MinScalarType(PyArrayObject* arr);

EIGENPY_DLLAPI int call_PyArray_RegisterCastFunc(
    PyArray_Descr* descr, int totype, PyArray_VectorUnaryFunc* castfunc);
#else
inline bool call_PyArray_Check(PyObject* py_obj) {
  return PyArray_Check(py_obj);
}

inline PyObject* call_PyArray_SimpleNew(int nd, npy_intp* shape, int np_type) {
  return PyArray_SimpleNew(nd, shape, np_type);
}

inline PyObject* call_PyArray_New(PyTypeObject* py_type_ptr, int nd,
                                  npy_intp* shape, int np_type, void* data_ptr,
                                  int options) {
  return PyArray_New(py_type_ptr, nd, shape, np_type, NULL, data_ptr, 0,
                     options, NULL);
}

inline PyObject* call_PyArray_New(PyTypeObject* py_type_ptr, int nd,
                                  npy_intp* shape, int np_type,
                                  npy_intp* strides, void* data_ptr,
                                  int options) {
  return PyArray_New(py_type_ptr, nd, shape, np_type, strides, data_ptr, 0,
                     options, NULL);
}

inline int call_PyArray_ObjectType(PyObject* obj, int val) {
  return PyArray_ObjectType(obj, val);
}

inline PyTypeObject* getPyArrayType() { return &PyArray_Type; }

inline PyArray_Descr* call_PyArray_DescrFromType(int typenum) {
  return PyArray_DescrFromType(typenum);
}

inline void call_PyArray_InitArrFuncs(PyArray_ArrFuncs* funcs) {
  PyArray_InitArrFuncs(funcs);
}

inline int call_PyArray_RegisterDataType(PyArray_DescrProto* dtype) {
  return PyArray_RegisterDataType(dtype);
}

inline PyArray_Descr* call_PyArray_MinScalarType(PyArrayObject* arr) {
  return PyArray_MinScalarType(arr);
}

inline int call_PyArray_RegisterCanCast(PyArray_Descr* descr, int totype,
                                        NPY_SCALARKIND scalar) {
  return PyArray_RegisterCanCast(descr, totype, scalar);
}

inline int call_PyArray_RegisterCastFunc(PyArray_Descr* descr, int totype,
                                         PyArray_VectorUnaryFunc* castfunc) {
  return PyArray_RegisterCastFunc(descr, totype, castfunc);
}
#endif
}  // namespace eigenpy

#endif  // ifndef __eigenpy_numpy_hpp__
