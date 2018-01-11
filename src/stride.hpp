/*
 * Copyright 2018, Justin Carpentier <jcarpent@laas.fr>, LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
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
