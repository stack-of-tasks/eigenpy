/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2022, INRIA
 */

#include <Eigen/Geometry>

#include "eigenpy/geometry.hpp"
#include "eigenpy/memory.hpp"

EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(Eigen::Quaterniond)

#include "eigenpy/quaternion.hpp"

namespace eigenpy {
void exposeQuaternion() { expose<Eigen::Quaterniond>(); }
}  // namespace eigenpy
