/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#ifndef __eigenpy_geometry_hpp__
#define __eigenpy_geometry_hpp__

#include "eigenpy/config.hpp"

namespace eigenpy
{
  
  void EIGENPY_DLLEXPORT exposeQuaternion();
  void EIGENPY_DLLEXPORT exposeAngleAxis();
  
  void EIGENPY_DLLEXPORT exposeGeometryConversion();
  
} // namespace eigenpy

#endif // define __eigenpy_geometry_hpp__
