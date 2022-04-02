/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixLong() {
  exposeType<long>();
  exposeType<long, Eigen::RowMajor>();
}
}  // namespace eigenpy
