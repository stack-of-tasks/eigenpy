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

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type,
          typename Scalar = typename EigenType::Scalar>
struct expose_eigen_type_impl;

template <typename MatType, typename Scalar>
struct expose_eigen_type_impl<MatType, Eigen::MatrixBase<MatType>, Scalar> {
  static void run() {
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
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType, typename Scalar>
struct expose_eigen_type_impl<TensorType, Eigen::TensorBase<TensorType>,
                              Scalar> {
  static void run() {
    if (check_registration<TensorType>()) return;

    // to-python
    EigenToPyConverter<TensorType>::registration();
    EigenToPyConverter<Eigen::TensorRef<TensorType> >::registration();
    EigenToPyConverter<
        const Eigen::TensorRef<const TensorType> >::registration();

    // from-python
    EigenFromPyConverter<TensorType>::registration();
  }
};
#endif

template <typename MatType>
void enableEigenPySpecific() {
  expose_eigen_type_impl<MatType>::run();
}

}  // namespace eigenpy

#endif  // ifndef __eigenpy_details_hpp__
