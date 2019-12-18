/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <stdlib.h>

namespace eigenpy
{

  void seed(unsigned int seed_value)
  {
    srand(seed_value);
  }

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

    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xi);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xi);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xi);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXf);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXi);

    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXf);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXi);

    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXf);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXi);
  }

} // namespace eigenpy
