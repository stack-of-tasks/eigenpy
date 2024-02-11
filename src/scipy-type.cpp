/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/scipy-type.hpp"

#include <patchlevel.h>  // For PY_MAJOR_VERSION

namespace eigenpy {

ScipyType& ScipyType::getInstance() {
  static ScipyType instance;
  return instance;
}

void ScipyType::sharedMemory(const bool value) {
  getInstance().shared_memory = value;
}

bool ScipyType::sharedMemory() { return getInstance().shared_memory; }

const PyTypeObject* ScipyType::getScipyCSRMatrixType() {
  return getInstance().csr_matrix_type;
}

const PyTypeObject* ScipyType::getScipyCSCMatrixType() {
  return getInstance().csc_matrix_type;
}

ScipyType::ScipyType() {
  try {
    sparse_module = bp::import("scipy.sparse");
  } catch (...) {
    throw std::runtime_error(
        "SciPy is not installed. "
        "You can install it using the command \'pip install scipy\'.");
  }

#if PY_MAJOR_VERSION >= 3
  // TODO I don't know why this Py_INCREF is necessary.
  // Without it, the destructor of ScipyType SEGV sometimes.
  Py_INCREF(sparse_module.ptr());
#endif

  csr_matrix_obj = sparse_module.attr("csr_matrix");
  csr_matrix_type = reinterpret_cast<PyTypeObject*>(csr_matrix_obj.ptr());
  csc_matrix_obj = sparse_module.attr("csc_matrix");
  csc_matrix_type = reinterpret_cast<PyTypeObject*>(csc_matrix_obj.ptr());

  shared_memory = true;
}
}  // namespace eigenpy
