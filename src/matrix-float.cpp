/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixFloat() {
  exposeType<float>();
  exposeType<float, Eigen::RowMajor>();
}
}  // namespace eigenpy
