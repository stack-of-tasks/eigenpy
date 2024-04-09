/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixInt16() {
  exposeType<int16_t>();
  exposeType<int16_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
