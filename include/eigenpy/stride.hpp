/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_stride_hpp__
#define __eigenpy_stride_hpp__

#include <Eigen/Core>

namespace eigenpy
{
  template<typename MatType, int IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
  struct StrideType
  {
    typedef Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> type;
  };
  
  template<typename MatType>
  struct StrideType<MatType,1>
  {
    typedef Eigen::InnerStride<Eigen::Dynamic> type;
  };
}

#endif // ifndef __eigenpy_stride_hpp__
