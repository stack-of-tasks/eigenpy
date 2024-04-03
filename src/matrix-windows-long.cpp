/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixWindowsLong() {
// On Windows, long is a 32 bytes type but it's a different type than int
#ifdef WIN32
  exposeType<long>();
  exposeType<long, Eigen::RowMajor>();
#endif  // WIN32
}
}  // namespace eigenpy
