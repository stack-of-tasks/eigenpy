/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixUInt16() {
  exposeType<uint16_t>();
  exposeType<uint16_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
