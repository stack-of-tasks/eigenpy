/*
 * Copyright 2020-2024 INRIA
 */

#include "eigenpy/numpy.hpp"

namespace eigenpy {
void import_numpy() {
  if (_import_array() < 0) {
    PyErr_Print();
    PyErr_SetString(PyExc_ImportError,
                    "numpy.core.multiarray failed to import");
  }
}

int PyArray_TypeNum(PyTypeObject* type) {
  PyArray_Descr* descr =
      PyArray_DescrFromTypeObject(reinterpret_cast<PyObject*>(type));
  if (descr == NULL) {
    return NPY_NOTYPE;
  }
  return descr->type_num;
}

#if defined _WIN32 || defined __CYGWIN__

bool call_PyArray_Check(PyObject* py_obj) { return PyArray_Check(py_obj); }

PyObject* call_PyArray_SimpleNew(int nd, npy_intp* shape, int np_type) {
  return PyArray_SimpleNew(nd, shape, np_type);
}

PyObject* call_PyArray_New(PyTypeObject* py_type_ptr, int nd, npy_intp* shape,
                           int np_type, void* data_ptr, int options) {
  return PyArray_New(py_type_ptr, nd, shape, np_type, NULL, data_ptr, 0,
                     options, NULL);
}

PyObject* call_PyArray_New(PyTypeObject* py_type_ptr, int nd, npy_intp* shape,
                           int np_type, npy_intp* strides, void* data_ptr,
                           int options) {
  return PyArray_New(py_type_ptr, nd, shape, np_type, strides, data_ptr, 0,
                     options, NULL);
}

int call_PyArray_ObjectType(PyObject* obj, int val) {
  return PyArray_ObjectType(obj, val);
}

PyTypeObject* getPyArrayType() { return &PyArray_Type; }

PyArray_Descr* call_PyArray_DescrFromType(int typenum) {
  return PyArray_DescrFromType(typenum);
}

void call_PyArray_InitArrFuncs(PyArray_ArrFuncs* funcs) {
  PyArray_InitArrFuncs(funcs);
}

int call_PyArray_RegisterDataType(PyArray_DescrProto* dtype) {
  return PyArray_RegisterDataType(dtype);
}

PyArray_Descr* call_PyArray_MinScalarType(PyArrayObject* arr) {
  return PyArray_MinScalarType(arr);
}

int call_PyArray_RegisterCanCast(PyArray_Descr* descr, int totype,
                                 NPY_SCALARKIND scalar) {
  return PyArray_RegisterCanCast(descr, totype, scalar);
}

int call_PyArray_RegisterCastFunc(PyArray_Descr* descr, int totype,
                                  PyArray_VectorUnaryFunc* castfunc) {
  return PyArray_RegisterCastFunc(descr, totype, castfunc);
}

#endif
}  // namespace eigenpy
