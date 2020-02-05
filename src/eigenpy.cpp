/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <stdlib.h>


namespace eigenpy
{

  void seed(unsigned int seed_value)
  {
    srand(seed_value);
  }

  void exposeMatrixInt();
  void exposeMatrixLong();
  void exposeMatrixFloat();
  void exposeMatrixDouble();
  void exposeMatrixLongDouble();

  void exposeMatrixComplexFloat();
  void exposeMatrixComplexDouble();
  void exposeMatrixComplexLongDouble();

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instances. */
  void enableEigenPy()
  {
    using namespace Eigen;
    
    Exception::registerException();
    
    bp::def("setNumpyType",&NumpyType::setNumpyType,
            bp::arg("Numpy type (np.ndarray or np.matrix)"),
            "Change the Numpy type returned by the converters from an Eigen object.");
            
    bp::def("getNumpyType",&NumpyType::getNumpyType,
            "Get the Numpy type returned by the converters from an Eigen object.");
    
    bp::def("switchToNumpyArray",&NumpyType::switchToNumpyArray,
            "Set the conversion from Eigen::Matrix to numpy.ndarray.");
    
    bp::def("switchToNumpyMatrix",&NumpyType::switchToNumpyMatrix,
            "Set the conversion from Eigen::Matrix to numpy.matrix.");
    
    bp::def("seed",&seed,bp::arg("seed_value"),
            "Initialize the pseudo-random number generator with the argument seed_value.");
    
    exposeMatrixInt();
    exposeMatrixLong();
    exposeMatrixFloat();
    exposeMatrixDouble();
    exposeMatrixLongDouble();
    
    exposeMatrixComplexFloat();
    exposeMatrixComplexDouble();
    exposeMatrixComplexLongDouble();
  }

} // namespace eigenpy
