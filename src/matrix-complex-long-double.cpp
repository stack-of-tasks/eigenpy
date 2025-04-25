/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixComplexLongDouble() {
  exposeType<std::complex<long double>>();
  exposeType<std::complex<long double>, Eigen::RowMajor>();
}
}  // namespace eigenpy
