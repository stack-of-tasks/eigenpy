/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy {
void exposeMatrixComplexDouble() {
  exposeType<std::complex<double>>();
  exposeType<std::complex<double>, Eigen::RowMajor>();
}
}  // namespace eigenpy
