/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_geometry_hpp__
#define __eigenpy_geometry_hpp__

#include "eigenpy/eigenpy_export.h"

namespace eigenpy
{
  
  void EIGENPY_EXPORT exposeQuaternion();
  void EIGENPY_EXPORT exposeAngleAxis();
  
  void EIGENPY_EXPORT exposeGeometryConversion();
  
} // namespace eigenpy

#endif // define __eigenpy_geometry_hpp__
