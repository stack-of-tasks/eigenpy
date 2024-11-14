/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixMacULong() {
// On Mac, long is a 64 bytes type but it's a different type than int64_t
#ifdef __APPLE__
  exposeType<unsigned long>();
  exposeType<unsigned long, Eigen::RowMajor>();
#endif  // Mac
}
}  // namespace eigenpy
