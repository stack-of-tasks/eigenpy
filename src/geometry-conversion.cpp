/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/memory.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/geometry-conversion.hpp"

namespace eigenpy
{
  void exposeGeometryConversion()
  {
    EulerAnglesConvertor<double>::expose();
  }
} // namespace eigenpy
