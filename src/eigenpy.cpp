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

  template<typename Scalar, int Options>
  EIGEN_DONT_INLINE void exposeType()
  {
    EIGENPY_MAKE_TYPEDEFS_ALL_SIZES(Scalar,Options,s);

    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2s);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2s);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2s);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xs);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2s);

    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3s);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3s);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3s);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xs);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3s);

    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4s);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4s);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4s);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xs);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4s);

    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXs);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXs);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXs);
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
    import_numpy();
    
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
    
    bp::def("sharedMemory",
            (void (*)(const bool))NumpyType::sharedMemory,
            bp::arg("value"),
            "Share the memory when converting from Eigen to Numpy.");
    
    bp::def("sharedMemory",
            (bool (*)())NumpyType::sharedMemory,
            "Status of the shared memory when converting from Eigen to Numpy.\n"
            "If True, the memory is shared when converting an Eigen::Matrix to a numpy.array.\n"
            "Otherwise, a deep copy of the Eigen::Matrix is performed.");
    
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
