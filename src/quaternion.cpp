/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/memory.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/quaternion.hpp"

EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(Eigen::Quaterniond)

namespace eigenpy
{
  void exposeQuaternion()
  {
    expose<Eigen::Quaterniond>();
  }
} // namespace eigenpy
