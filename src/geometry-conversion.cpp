/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include "eigenpy/geometry-conversion.hpp"
#include "eigenpy/geometry.hpp"

namespace eigenpy {
void exposeGeometryConversion() { EulerAnglesConvertor<double>::expose(); }
}  // namespace eigenpy
