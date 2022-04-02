/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/angle-axis.hpp"

#include "eigenpy/geometry.hpp"
#include "eigenpy/memory.hpp"

EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(Eigen::AngleAxisd)

namespace eigenpy {
void exposeAngleAxis() { expose<Eigen::AngleAxisd>(); }
}  // namespace eigenpy
