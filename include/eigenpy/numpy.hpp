/*
 * Copyright 2020-2022 INRIA
 */

#ifndef __eigenpy_numpy_hpp__
#define __eigenpy_numpy_hpp__

#include "eigenpy/fwd.hpp"

#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL EIGENPY_ARRAY_API
#endif

#include <numpy/numpyconfig.h>
#ifdef NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#if defined _WIN32 || defined __CYGWIN__
#define EIGENPY_GET_PY_ARRAY_TYPE(array) \
  call_PyArray_MinScalarType(array)->type_num
#else
#define EIGENPY_GET_PY_ARRAY_TYPE(array) PyArray_MinScalarType(array)->type_num
#endif

namespace eigenpy {
void EIGENPY_DLLAPI import_numpy();
int EIGENPY_DLLAPI PyArray_TypeNum(PyTypeObject* type);

// By default, the Scalar is considered as a Python object
template <typename Scalar>
struct NumpyEquivalentType {
  enum { type_code = NPY_USERDEF };
};

template <>
struct NumpyEquivalentType<float> {
  enum { type_code = NPY_FLOAT };
};
template <>
struct NumpyEquivalentType<std::complex<float> > {
  enum { type_code = NPY_CFLOAT };
};
template <>
struct NumpyEquivalentType<double> {
  enum { type_code = NPY_DOUBLE };
};
template <>
struct NumpyEquivalentType<std::complex<double> > {
  enum { type_code = NPY_CDOUBLE };
};
template <>
struct NumpyEquivalentType<long double> {
  enum { type_code = NPY_LONGDOUBLE };
};
template <>
struct NumpyEquivalentType<std::complex<long double> > {
  enum { type_code = NPY_CLONGDOUBLE };
};
template <>
struct NumpyEquivalentType<bool> {
  enum { type_code = NPY_BOOL };
};
template <>
struct NumpyEquivalentType<int> {
  enum { type_code = NPY_INT };
};
template <>
struct NumpyEquivalentType<unsigned int> {
  enum { type_code = NPY_UINT };
};
template <>
struct NumpyEquivalentType<long> {
  enum { type_code = NPY_LONG };
};
// #if defined _WIN32 || defined __CYGWIN__
template <>
struct NumpyEquivalentType<long long> {
  enum { type_code = NPY_LONGLONG };
};
// #else
//   template <> struct NumpyEquivalentType<long long>    { enum { type_code =
//   NPY_LONGLONG    };};
// #endif
template <>
struct NumpyEquivalentType<unsigned long> {
  enum { type_code = NPY_ULONG };
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

EIGENPY_DLLAPI int call_PyArray_RegisterDataType(PyArray_Descr* dtype);

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

inline int call_PyArray_RegisterDataType(PyArray_Descr* dtype) {
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
