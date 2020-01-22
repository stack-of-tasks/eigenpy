/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include <stdlib.h>

namespace Eigen
{
  #define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
  /** \ingroup matrixtypedefs */                                    \
  typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
  /** \ingroup matrixtypedefs */                                    \
  typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;  \
  /** \ingroup matrixtypedefs */                                    \
  typedef Matrix<Type, 1, Size>    RowVector##SizeSuffix##TypeSuffix;

  #define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
  /** \ingroup matrixtypedefs */                                    \
  typedef Matrix<Type, Size, Dynamic> Matrix##Size##X##TypeSuffix;  \
  /** \ingroup matrixtypedefs */                                    \
  typedef Matrix<Type, Dynamic, Size> Matrix##X##Size##TypeSuffix;

  #define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X) \
  EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
  EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
  EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

  EIGEN_MAKE_TYPEDEFS_ALL_SIZES(long double,               ld)
  EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<long double>, cld)

  #undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
  #undef EIGEN_MAKE_TYPEDEFS
  #undef EIGEN_MAKE_FIXED_TYPEDEFS
}

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
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xi);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xcf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xcd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix2Xcld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xi);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xcf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xcd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix3Xcld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xi);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xcf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xcd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Matrix4Xcld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX2cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX3cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixX4cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXld);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXf);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXi);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXcf);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXcd);
    ENABLE_SPECIFIC_MATRIX_TYPE(MatrixXcld);

    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector2cd);

    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector3cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVector4cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXld);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXf);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXi);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXcf);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXcd);
    ENABLE_SPECIFIC_MATRIX_TYPE(RowVectorXcld);

    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector2cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector3cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4ld);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4d);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4f);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4i);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4cf);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4cd);
    ENABLE_SPECIFIC_MATRIX_TYPE(Vector4cld);
    
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXld);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXd);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXf);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXi);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXcf);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXcd);
    ENABLE_SPECIFIC_MATRIX_TYPE(VectorXcld);
  }

} // namespace eigenpy
