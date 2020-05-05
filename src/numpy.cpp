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

  int PyArray_TypeNum(PyTypeObject * type)
  {
    return PyArray_TypeNumFromName(const_cast<char*>(type->tp_name));
  }

#if defined _WIN32 || defined __CYGWIN__

  PyObject* call_PyArray_SimpleNew(int nd, npy_intp * shape, int np_type)
  {
    return PyArray_SimpleNew(nd,shape,np_type);
  }

  PyObject* call_PyArray_New(PyTypeObject * py_type_ptr, int nd, npy_intp * shape, int np_type, void * data_ptr, int options)
  {
    return PyArray_New(py_type_ptr,nd,shape,np_type,NULL,data_ptr,0,options,NULL);
  }
  
  int call_PyArray_ObjectType(PyObject * obj, int val)
  {
    return PyArray_ObjectType(obj,val);
  }

  PyTypeObject * getPyArrayType() { return &PyArray_Type; }

  int call_PyArray_DescrFromType(int typenum)
  {
    return PyArray_DescrFromType(typenum);
  }

  void call_PyArray_InitArrFuncs(PyArray_ArrFuncs * funcs)
  {
    PyArray_InitArrFuncs(funcs);
  }

  int call_PyArray_RegisterDataType(PyArray_Descr * dtype)
  {
    return PyArray_RegisterDataType(dtype);
  }
  
#endif
}
