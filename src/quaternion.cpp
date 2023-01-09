/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include <Eigen/Geometry>

#include "eigenpy/geometry.hpp"
#include "eigenpy/quaternion.hpp"

namespace eigenpy {
void exposeQuaternion() { expose<Eigen::Quaterniond>(); }
}  // namespace eigenpy
