/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#ifndef __eigenpy_details_hpp__
#define __eigenpy_details_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/eigen-from-python.hpp"
#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/exception.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/registration.hpp"
#include "eigenpy/scalar-conversion.hpp"

namespace eigenpy {
template <typename MatType, typename EigenEquivalentType>
EIGENPY_DEPRECATED void enableEigenPySpecific() {
  enableEigenPySpecific<MatType>();
}

template <typename MatType>
void enableEigenPySpecific() {
  if (check_registration<MatType>()) return;

  // to-python
  EigenToPyConverter<MatType>::registration();
#if EIGEN_VERSION_AT_LEAST(3, 2, 0)
  EigenToPyConverter<Eigen::Ref<MatType> >::registration();
  EigenToPyConverter<const Eigen::Ref<const MatType> >::registration();
#endif

  // from-python
  EigenFromPyConverter<MatType>::registration();
}

}  // namespace eigenpy

#endif  // ifndef __eigenpy_details_hpp__
