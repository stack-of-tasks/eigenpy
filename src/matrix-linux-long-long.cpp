/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixLinuxLongLong() {
// On Linux, long long is a 64 bytes type but it's a different type than int64_t
#ifdef __linux__
  exposeType<long long>();
  exposeType<long long, Eigen::RowMajor>();
#endif  // linux
}
}  // namespace eigenpy
