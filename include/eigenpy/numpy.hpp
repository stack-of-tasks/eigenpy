/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_numpy_hpp__
#define __eigenpy_numpy_hpp__

#include <boost/python.hpp>
#include "eigenpy/config.hpp"

#ifndef PY_ARRAY_UNIQUE_SYMBOL
  #define PY_ARRAY_UNIQUE_SYMBOL EIGENPY_ARRAY_API
#endif

#include <numpy/numpyconfig.h>
#ifdef NPY_1_8_API_VERSION
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/noprefix.h>
#include <numpy/ufuncobject.h>

#if defined _WIN32 || defined __CYGWIN__
  #define EIGENPY_GET_PY_ARRAY_TYPE(array) \
    call_PyArray_ObjectType(reinterpret_cast<PyObject *>(array), 0)
#else
  #define EIGENPY_GET_PY_ARRAY_TYPE(array) \
    PyArray_ObjectType(reinterpret_cast<PyObject *>(array), 0)
#endif

namespace eigenpy
{
  void EIGENPY_DLLAPI import_numpy();
  int EIGENPY_DLLAPI PyArray_TypeNum(PyTypeObject * type);
}

#if defined _WIN32 || defined __CYGWIN__
namespace eigenpy
{
  EIGENPY_DLLAPI bool call_PyArray_Check(PyObject *);

  EIGENPY_DLLAPI PyObject* call_PyArray_SimpleNew(int nd, npy_intp * shape, int np_type);

  EIGENPY_DLLAPI PyObject* call_PyArray_New(PyTypeObject * py_type_ptr, int nd, npy_intp * shape, int np_type, void * data_ptr, int options);

  EIGENPY_DLLAPI int call_PyArray_ObjectType(PyObject *, int);

  EIGENPY_DLLAPI PyTypeObject * getPyArrayType();

  EIGENPY_DLLAPI PyArray_Descr * call_PyArray_DescrFromType(int typenum);

  EIGENPY_DLLAPI void call_PyArray_InitArrFuncs(PyArray_ArrFuncs * funcs);

  EIGENPY_DLLAPI int call_PyArray_RegisterDataType(PyArray_Descr * dtype);
}
#else
  #define call_PyArray_Check(py_obj) PyArray_Check(py_obj)
  #define call_PyArray_SimpleNew PyArray_SimpleNew
  #define call_PyArray_New(py_type_ptr,nd,shape,np_type,data_ptr,options) \
    PyArray_New(py_type_ptr,nd,shape,np_type,NULL,data_ptr,0,options,NULL)
  #define getPyArrayType() &PyArray_Type
  #define call_PyArray_DescrFromType(typenum) PyArray_DescrFromType(typenum)
  #define call_PyArray_InitArrFuncs(funcs) PyArray_InitArrFuncs(funcs)
 #define call_PyArray_RegisterDataType(dtype) PyArray_RegisterDataType(dtype)
#endif

#endif // ifndef __eigenpy_numpy_hpp__
