/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_stride_hpp__
#define __eigenpy_stride_hpp__

#include <Eigen/Core>

namespace eigenpy
{
  template<typename MatType, int InnerStride = Eigen::Dynamic, int OuterStride = Eigen::Dynamic, bool IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
  struct StrideType
  {
    typedef Eigen::Stride<OuterStride,InnerStride> type;
  };
  
  template<typename MatType, int InnerStride, int OuterStride>
  struct StrideType<MatType,InnerStride,OuterStride,true>
  {
    typedef Eigen::InnerStride<InnerStride> type;
  };

}

#endif // ifndef __eigenpy_stride_hpp__
