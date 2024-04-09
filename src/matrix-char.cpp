/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <cstdint>

namespace eigenpy {
void exposeMatrixChar() {
  exposeType<char>();
  exposeType<char, Eigen::RowMajor>();
}
}  // namespace eigenpy
