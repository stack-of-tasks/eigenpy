/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixInt8() {
  exposeType<int8_t>();
  exposeType<int8_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
