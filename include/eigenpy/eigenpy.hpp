/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_eigenpy_hpp__
#define __eigenpy_eigenpy_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/deprecated.hh"
#include "eigenpy/eigenpy_export.h"

#if EIGEN_VERSION_AT_LEAST(3,2,0)
#include "eigenpy/ref.hpp"

#define ENABLE_SPECIFIC_MATRIX_TYPE(TYPE) \
  ::eigenpy::enableEigenPySpecific<TYPE>(); \
  ::eigenpy::enableEigenPySpecific< eigenpy::Ref<TYPE> >();

#else // if EIGEN_VERSION_AT_LEAST(3,2,0)

#define ENABLE_SPECIFIC_MATRIX_TYPE(TYPE) \
  ::eigenpy::enableEigenPySpecific<TYPE>();

#endif // if EIGEN_VERSION_AT_LEAST(3,2,0)

namespace eigenpy
{

  /* Load numpy through Python */
  void loadNumpyArray();

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance. */
  void EIGENPY_EXPORT enableEigenPy();

  template<typename MatType>
  void enableEigenPySpecific();
  
  /* Enable the Eigen--Numpy serialization for the templated MatrixBase class.
   * The second template argument is used for inheritance of Eigen classes. If
   * using a native Eigen::MatrixBase, simply repeat the same arg twice. */
  template<typename MatType,typename EigenEquivalentType>
  EIGENPY_DEPRECATED void enableEigenPySpecific();


} // namespace eigenpy

#include "eigenpy/details.hpp"

#endif // ifndef __eigenpy_eigenpy_hpp__

