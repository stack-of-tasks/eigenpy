/*
 * Copyright 2021 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixBool() {
  exposeType<bool>();
  exposeType<bool, Eigen::RowMajor>();
}
}  // namespace eigenpy
