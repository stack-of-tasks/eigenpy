/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#ifndef __eigenpy_stride_hpp__
#define __eigenpy_stride_hpp__

#include <eigenpy/fwd.hpp>

namespace eigenpy {

template <typename MatType, int InnerStride, int OuterStride,
          bool IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
struct stride_type_matrix {
  typedef Eigen::Stride<OuterStride, InnerStride> type;
};

template <typename MatType, int InnerStride, int OuterStride>
struct stride_type_matrix<MatType, InnerStride, OuterStride, true> {
  typedef Eigen::InnerStride<InnerStride> type;
};

template <typename EigenType, int InnerStride, int OuterStride,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct stride_type;

template <typename MatrixType, int InnerStride, int OuterStride>
struct stride_type<MatrixType, InnerStride, OuterStride,
                   Eigen::MatrixBase<MatrixType> > {
  typedef
      typename stride_type_matrix<MatrixType, InnerStride, OuterStride>::type
          type;
};

template <typename MatrixType, int InnerStride, int OuterStride>
struct stride_type<const MatrixType, InnerStride, OuterStride,
                   const Eigen::MatrixBase<MatrixType> > {
  typedef typename stride_type_matrix<const MatrixType, InnerStride,
                                      OuterStride>::type type;
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType, int InnerStride, int OuterStride>
struct stride_type<TensorType, InnerStride, OuterStride,
                   Eigen::TensorBase<TensorType> > {
  typedef Eigen::Stride<OuterStride, InnerStride> type;
};

template <typename TensorType, int InnerStride, int OuterStride>
struct stride_type<const TensorType, InnerStride, OuterStride,
                   const Eigen::TensorBase<TensorType> > {
  typedef Eigen::Stride<OuterStride, InnerStride> type;
};
#endif

template <typename EigenType, int InnerStride = Eigen::Dynamic,
          int OuterStride = Eigen::Dynamic>
struct StrideType {
  typedef typename stride_type<EigenType, InnerStride, OuterStride>::type type;
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_stride_hpp__
