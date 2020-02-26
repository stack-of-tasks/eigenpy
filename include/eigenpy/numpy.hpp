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

#if defined _WIN32 || defined __CYGWIN__
  #define EIGENPY_GET_PY_ARRAY_TYPE(array) \
    call_PyArray_ObjectType(reinterpret_cast<PyObject *>(array), 0)
#else
  #define EIGENPY_GET_PY_ARRAY_TYPE(array) \
    PyArray_ObjectType(reinterpret_cast<PyObject *>(array), 0)
#endif

namespace eigenpy
{
  void EIGENPY_DLLEXPORT import_numpy();
}

#if defined _WIN32 || defined __CYGWIN__
namespace eigenpy
{
  EIGENPY_DLLEXPORT PyObject*  call_PyArray_SimpleNew(int nd, npy_intp * shape, NPY_TYPES np_type);

  EIGENPY_DLLEXPORT PyObject* call_PyArray_New(int nd, npy_intp * shape, NPY_TYPES np_type, void * data_ptr, npy_intp options);

  EIGENPY_DLLEXPORT int call_PyArray_ObjectType(PyObject *, int);
}
#else
  #define call_PyArray_SimpleNew PyArray_SimpleNew
  #define call_PyArray_New(nd,shape,np_type,data_ptr,options) \
    PyArray_New(&PyArray_Type,nd,shape,np_type,NULL,data_ptr,0,options,NULL)
#endif

#endif // ifndef __eigenpy_numpy_hpp__
