/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include "eigenpy/eigenpy.hpp"

#include <stdlib.h>

namespace eigenpy {

void seed(unsigned int seed_value) { srand(seed_value); }

void exposeMatrixBool();
void exposeMatrixInt();
void exposeMatrixLong();
void exposeMatrixFloat();
void exposeMatrixDouble();
void exposeMatrixLongDouble();

void exposeMatrixComplexFloat();
void exposeMatrixComplexDouble();
void exposeMatrixComplexLongDouble();

void exposeNoneType();

/* Enable Eigen-Numpy serialization for a set of standard MatrixBase instances.
 */
void enableEigenPy() {
  using namespace Eigen;
  import_numpy();

  Exception::registerException();

  bp::def("sharedMemory", (void (*)(const bool))NumpyType::sharedMemory,
          bp::arg("value"),
          "Share the memory when converting from Eigen to Numpy.");

  bp::def("sharedMemory", (bool (*)())NumpyType::sharedMemory,
          "Status of the shared memory when converting from Eigen to Numpy.\n"
          "If True, the memory is shared when converting an Eigen::Matrix to a "
          "numpy.array.\n"
          "Otherwise, a deep copy of the Eigen::Matrix is performed.");

  bp::def("seed", &seed, bp::arg("seed_value"),
          "Initialize the pseudo-random number generator with the argument "
          "seed_value.");

  exposeMatrixBool();
  exposeMatrixInt();
  exposeMatrixLong();
  exposeMatrixFloat();
  exposeMatrixDouble();
  exposeMatrixLongDouble();

  exposeMatrixComplexFloat();
  exposeMatrixComplexDouble();
  exposeMatrixComplexLongDouble();

  exposeNoneType();
}

bool withTensorSupport() {
#ifdef EIGENPY_WITH_TENSOR_SUPPORT
  return true;
#else
  return false;
#endif
}

}  // namespace eigenpy
