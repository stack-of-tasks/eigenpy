/*
* Copyright 2020 INRIA
*/

#ifndef __eigenpy_utils_scalar_is_approx_hpp__
#define __eigenpy_utils_scalar_is_approx_hpp__

#include <Eigen/Core>

namespace eigenpy
{
  template<typename MatrixType1, typename MatrixType2>
  inline bool is_approx(const Eigen::MatrixBase<MatrixType1> & mat1,
                        const Eigen::MatrixBase<MatrixType2> & mat2,
                        const typename MatrixType1::Scalar & prec = Eigen::NumTraits<typename MatrixType1::Scalar>::dummy_precision())
  {
    return mat1.isApprox(mat2,prec);
  }
}

#endif // ifndef __eigenpy_utils_scalar_is_approx_hpp__
