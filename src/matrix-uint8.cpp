/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixUInt8() {
  exposeType<std::uint8_t>();
  exposeType<std::uint8_t, Eigen::RowMajor>();
}
}  // namespace eigenpy
