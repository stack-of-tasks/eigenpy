/*
 * Copyright 2018-2023 INRIA
 */

#include "eigenpy/numpy-type.hpp"

#include <patchlevel.h>  // For PY_MAJOR_VERSION

namespace eigenpy {

NumpyType& NumpyType::getInstance() {
  static NumpyType instance;
  return instance;
}

bp::object NumpyType::make(PyArrayObject* pyArray, bool copy) {
  return make((PyObject*)pyArray, copy);
}

bp::object NumpyType::make(PyObject* pyObj, bool /*copy*/) {
  bp::object m;
  m = bp::object(bp::handle<>(pyObj));  // nothing to do here

  Py_INCREF(m.ptr());
  return m;
}

void NumpyType::sharedMemory(const bool value) {
  getInstance().shared_memory = value;
}

bool NumpyType::sharedMemory() { return getInstance().shared_memory; }

const PyTypeObject* NumpyType::getNumpyArrayType() {
  return getInstance().NumpyArrayType;
}

NumpyType::NumpyType() {
  pyModule = bp::import("numpy");

#if PY_MAJOR_VERSION >= 3
  // TODO I don't know why this Py_INCREF is necessary.
  // Without it, the destructor of NumpyType SEGV sometimes.
  Py_INCREF(pyModule.ptr());
#endif

  NumpyArrayObject = pyModule.attr("ndarray");
  NumpyArrayType = reinterpret_cast<PyTypeObject*>(NumpyArrayObject.ptr());

  shared_memory = true;
}
}  // namespace eigenpy
