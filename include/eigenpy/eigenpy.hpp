/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#ifndef __eigenpy_eigenpy_hpp__
#define __eigenpy_eigenpy_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-typedef.hpp"
#include "eigenpy/expose.hpp"

#define ENABLE_SPECIFIC_MATRIX_TYPE(TYPE) \
  ::eigenpy::enableEigenPySpecific<TYPE>();

namespace eigenpy {

/* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance.
 */
void EIGENPY_DLLAPI enableEigenPy();

/* Enable the Eigen--Numpy serialization for the templated MatType class.*/
template <typename MatType>
void enableEigenPySpecific();

template <typename Scalar, int Options>
EIGEN_DONT_INLINE void exposeType() {
  EIGENPY_MAKE_TYPEDEFS_ALL_SIZES(Scalar, Options, s);

  EIGENPY_UNUSED_TYPE(Vector1s);
  EIGENPY_UNUSED_TYPE(RowVector1s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix1s);

  ENABLE_SPECIFIC_MATRIX_TYPE(Vector2s);
  ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xs);
  ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2s);

  ENABLE_SPECIFIC_MATRIX_TYPE(Vector3s);
  ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xs);
  ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3s);

  ENABLE_SPECIFIC_MATRIX_TYPE(Vector4s);
  ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4s);
  ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xs);
  ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4s);

  ENABLE_SPECIFIC_MATRIX_TYPE(VectorXs);
  ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXs);
  ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXs);
}

template <typename Scalar>
EIGEN_DONT_INLINE void exposeType() {
  exposeType<Scalar, 0>();

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
  enableEigenPySpecific<Eigen::Tensor<Scalar, 1> >();
  enableEigenPySpecific<Eigen::Tensor<Scalar, 2> >();
  enableEigenPySpecific<Eigen::Tensor<Scalar, 3> >();
#endif
}

}  // namespace eigenpy

#include "eigenpy/details.hpp"

#endif  // ifndef __eigenpy_eigenpy_hpp__
