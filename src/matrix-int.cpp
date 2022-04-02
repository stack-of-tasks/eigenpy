/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixInt() {
  exposeType<int>();
  exposeType<int, Eigen::RowMajor>();
}
}  // namespace eigenpy
