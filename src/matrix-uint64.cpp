/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixUInt64() {
  exposeType<uint64_t>();
  exposeType<uint64_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
