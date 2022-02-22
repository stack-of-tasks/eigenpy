/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2022, INRIA
 */

#include "eigenpy/memory.hpp"
#include "eigenpy/geometry.hpp"

#include <Eigen/Geometry>

EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(Eigen::Quaterniond)

#include "eigenpy/quaternion.hpp"

namespace eigenpy
{
  void exposeQuaternion()
  {
    expose<Eigen::Quaterniond>();
  }
} // namespace eigenpy
