/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_scipy_type_hpp__
#define __eigenpy_scipy_type_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/register.hpp"
#include "eigenpy/scalar-conversion.hpp"
#include "eigenpy/numpy-type.hpp"

namespace eigenpy {

struct EIGENPY_DLLAPI ScipyType {
  static ScipyType& getInstance();

  static void sharedMemory(const bool value);

  static bool sharedMemory();

  static bp::object getScipyType();

  static const PyTypeObject* getScipyCSRMatrixType();
  static const PyTypeObject* getScipyCSCMatrixType();

  template <typename SparseMatrix>
  static bp::object get_pytype_object(
      const Eigen::SparseMatrixBase<SparseMatrix>* ptr = nullptr) {
    EIGENPY_UNUSED_VARIABLE(ptr);
    return SparseMatrix::IsRowMajor ? getInstance().csr_matrix_obj
                                    : getInstance().csc_matrix_obj;
  }

 protected:
  ScipyType();

  bp::object sparse_module;

  // SciPy types
  bp::object csr_matrix_obj, csc_matrix_obj;
  PyTypeObject *csr_matrix_type, *csc_matrix_type;

  bool shared_memory;
};
}  // namespace eigenpy

#endif  // ifndef __eigenpy_scipy_type_hpp__
