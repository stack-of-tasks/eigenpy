/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace eigenpy
{

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instances. */
  void enableEigenPy()
  {
    using namespace Eigen;
    Exception::registerException();
    
    bp::def("setNumpyType",&NumpyType::setNumpyType,
            bp::arg("Numpy type (np.ndarray or np.matrix)"),
            "Change the type returned by the converters from an Eigen object.");
    
    bp::def("switchToNumpyArray",&NumpyType::switchToNumpyArray,
            "Set the conversion from Eigen::Matrix to numpy.ndarray.");
    
    bp::def("switchToNumpyMatrix",&NumpyType::switchToNumpyMatrix,
            "Set the conversion from Eigen::Matrix to numpy.matrix.");

    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4d);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4d);
  }

} // namespace eigenpy
