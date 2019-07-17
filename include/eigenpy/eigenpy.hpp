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
  enableEigenPySpecific<TYPE>(); \
  enableEigenPySpecific< eigenpy::Ref<TYPE> >();

#else

#define ENABLE_SPECIFIC_MATRIX_TYPE(TYPE) \
  enableEigenPySpecific<TYPE>();

#endif

namespace eigenpy
{
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

