/*
 * Copyright 2018-2020 INRIA
 */

#include "eigenpy/numpy-type.hpp"

#include <patchlevel.h>  // For PY_MAJOR_VERSION

namespace eigenpy {
namespace bp = boost::python;

NumpyType& NumpyType::getInstance() {
  static NumpyType instance;
  return instance;
}

bp::object NumpyType::make(PyArrayObject* pyArray, bool copy) {
  return make((PyObject*)pyArray, copy);
}

bp::object NumpyType::make(PyObject* pyObj, bool copy) {
  bp::object m;
  if (isMatrix())
    m = getInstance().NumpyMatrixObject(bp::object(bp::handle<>(pyObj)),
                                        bp::object(), copy);
  //    m = NumpyAsMatrixObject(bp::object(bp::handle<>(pyObj)));
  else if (isArray())
    m = bp::object(bp::handle<>(pyObj));  // nothing to do here

  Py_INCREF(m.ptr());
  return m;
}

void NumpyType::setNumpyType(bp::object& obj) {
  PyTypeObject* obj_type = PyType_Check(obj.ptr())
                               ? reinterpret_cast<PyTypeObject*>(obj.ptr())
                               : obj.ptr()->ob_type;
  if (PyType_IsSubtype(obj_type, getInstance().NumpyMatrixType))
    switchToNumpyMatrix();
  else if (PyType_IsSubtype(obj_type, getInstance().NumpyArrayType))
    switchToNumpyArray();
}

void NumpyType::sharedMemory(const bool value) {
  getInstance().shared_memory = value;
}

bool NumpyType::sharedMemory() { return getInstance().shared_memory; }

void NumpyType::switchToNumpyArray() {
  getInstance().CurrentNumpyType = getInstance().NumpyArrayObject;
  getInstance().getType() = ARRAY_TYPE;
}

void NumpyType::switchToNumpyMatrix() {
  getInstance().CurrentNumpyType = getInstance().NumpyMatrixObject;
  getInstance().getType() = MATRIX_TYPE;
}

NP_TYPE& NumpyType::getType() { return getInstance().np_type; }

bp::object NumpyType::getNumpyType() { return getInstance().CurrentNumpyType; }

const PyTypeObject* NumpyType::getNumpyMatrixType() {
  return getInstance().NumpyMatrixType;
}

const PyTypeObject* NumpyType::getNumpyArrayType() {
  return getInstance().NumpyArrayType;
}

bool NumpyType::isMatrix() {
  return PyType_IsSubtype(
      reinterpret_cast<PyTypeObject*>(getInstance().CurrentNumpyType.ptr()),
      getInstance().NumpyMatrixType);
}

bool NumpyType::isArray() {
  if (getInstance().isMatrix()) return false;
  return PyType_IsSubtype(
      reinterpret_cast<PyTypeObject*>(getInstance().CurrentNumpyType.ptr()),
      getInstance().NumpyArrayType);
}

NumpyType::NumpyType() {
  pyModule = bp::import("numpy");

#if PY_MAJOR_VERSION >= 3
  // TODO I don't know why this Py_INCREF is necessary.
  // Without it, the destructor of NumpyType SEGV sometimes.
  Py_INCREF(pyModule.ptr());
#endif

  NumpyMatrixObject = pyModule.attr("matrix");
  NumpyMatrixType = reinterpret_cast<PyTypeObject*>(NumpyMatrixObject.ptr());
  NumpyArrayObject = pyModule.attr("ndarray");
  NumpyArrayType = reinterpret_cast<PyTypeObject*>(NumpyArrayObject.ptr());
  // NumpyAsMatrixObject = pyModule.attr("asmatrix");
  // NumpyAsMatrixType =
  // reinterpret_cast<PyTypeObject*>(NumpyAsMatrixObject.ptr());

  CurrentNumpyType = NumpyArrayObject;  // default conversion
  np_type = ARRAY_TYPE;

  shared_memory = true;
}
}  // namespace eigenpy
