//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_swig_hpp__
#define __eigenpy_swig_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {
struct PySwigObject {
  PyObject_HEAD void* ptr;
  const char* desc;
};

inline PySwigObject* get_PySwigObject(PyObject* pyObj) {
  if (!PyObject_HasAttrString(pyObj, "this")) return NULL;

  PyObject* this_ptr = PyObject_GetAttrString(pyObj, "this");
  if (this_ptr == NULL) return NULL;
  PySwigObject* swig_obj = reinterpret_cast<PySwigObject*>(this_ptr);

  return swig_obj;
}
}  // namespace eigenpy

#endif  // ifndef __eigenpy_swig_hpp__
