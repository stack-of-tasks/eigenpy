/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixInt64() {
  exposeType<int64_t>();
  exposeType<int64_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
