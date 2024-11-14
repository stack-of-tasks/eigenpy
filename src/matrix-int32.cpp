/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixInt32() {
  exposeType<int32_t>();
  exposeType<int32_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
