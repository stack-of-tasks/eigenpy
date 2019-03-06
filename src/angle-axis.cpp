/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/memory.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/angle-axis.hpp"

EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(Eigen::AngleAxisd)

namespace eigenpy
{
  void exposeAngleAxis()
  {
    AngleAxisVisitor<Eigen::AngleAxisd>::expose();
  }
} // namespace eigenpy
