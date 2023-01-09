/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include "eigenpy/angle-axis.hpp"
#include "eigenpy/geometry.hpp"

namespace eigenpy {
void exposeAngleAxis() { expose<Eigen::AngleAxisd>(); }
}  // namespace eigenpy
