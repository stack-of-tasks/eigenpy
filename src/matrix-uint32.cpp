/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixUInt32() {
  exposeType<uint32_t>();
  exposeType<uint32_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
