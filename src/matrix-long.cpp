/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixLong() {
#ifdef WIN32
  exposeType<__int64>();
  exposeType<__int64, Eigen::RowMajor>();
#else
  exposeType<long>();
  exposeType<long, Eigen::RowMajor>();
#endif
}
}  // namespace eigenpy
