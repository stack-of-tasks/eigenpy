/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixDouble() {
  exposeType<double>();
  exposeType<double, Eigen::RowMajor>();
}
}  // namespace eigenpy
