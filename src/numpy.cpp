/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/numpy.hpp"

namespace eigenpy
{
  void import_numpy()
  {
    if(_import_array() < 0)
    {
      PyErr_Print();
      PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    }
  }

#if defined _WIN32 || defined __CYGWIN__

  PyArrayObject* call_PyArray_SimpleNew(npy_intp nd, npy_intp * shape, NPY_TYPES np_type)
  {
    return PyArray_SimpleNew(nd,shape,np_type);
  }

  PyArrayObject* call_PyArray_New(npy_intp nd, npy_intp * shape, NPY_TYPES np_type, void * data_ptr, npy_intp options)
  {
    return PyArray_New(&PyArray_Type,nd,shape,np_type,NULL,data_ptr,0,options,NULL);
  }
  
  int call_PyArray_ObjectType(PyObject * obj, int val)
  {
    return PyArray_ObjectType(obj,val);
  }

#endif
}
