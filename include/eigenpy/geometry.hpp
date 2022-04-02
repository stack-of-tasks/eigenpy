/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#ifndef __eigenpy_geometry_hpp__
#define __eigenpy_geometry_hpp__

#include "eigenpy/config.hpp"

namespace eigenpy {

void EIGENPY_DLLAPI exposeQuaternion();
void EIGENPY_DLLAPI exposeAngleAxis();

void EIGENPY_DLLAPI exposeGeometryConversion();

}  // namespace eigenpy

#endif  // define __eigenpy_geometry_hpp__
