/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#ifndef __eigenpy_eigenpy_hpp__
#define __eigenpy_eigenpy_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/deprecated.hpp"
#include "eigenpy/config.hpp"

#define ENABLE_SPECIFIC_MATRIX_TYPE(TYPE) \
  ::eigenpy::enableEigenPySpecific<TYPE>();

#define EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, Size, SizeSuffix)   \
  /** \ingroup matrixtypedefs */                                    \
  typedef Eigen::Matrix<Type, Size, Size, Options> Matrix##SizeSuffix##TypeSuffix;  \
  /** \ingroup matrixtypedefs */                                    \
  typedef Eigen::Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;  \
  /** \ingroup matrixtypedefs */                                    \
  typedef Eigen::Matrix<Type, 1, Size>    RowVector##SizeSuffix##TypeSuffix;

#define EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, Size)         \
  /** \ingroup matrixtypedefs */                                    \
  typedef Eigen::Matrix<Type, Size, Eigen::Dynamic, Options> Matrix##Size##X##TypeSuffix;  \
  /** \ingroup matrixtypedefs */                                    \
  typedef Eigen::Matrix<Type, Eigen::Dynamic, Size, Options> Matrix##X##Size##TypeSuffix;

#define EIGENPY_MAKE_TYPEDEFS_ALL_SIZES(Type, Options, TypeSuffix) \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 2, 2) \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 3, 3) \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 4, 4) \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, Eigen::Dynamic, X) \
  EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, 2) \
  EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, 3) \
  EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, 4)

namespace eigenpy
{

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance. */
  void EIGENPY_DLLAPI enableEigenPy();

  /* Enable the Eigen--Numpy serialization for the templated MatrixBase class.
   * The second template argument is used for inheritance of Eigen classes. If
   * using a native Eigen::MatrixBase, simply repeat the same arg twice. */
  template<typename MatType>
  void enableEigenPySpecific();
  
  /* Enable the Eigen--Numpy serialization for the templated MatrixBase class.
   * The second template argument is used for inheritance of Eigen classes. If
   * using a native Eigen::MatrixBase, simply repeat the same arg twice. */
  template<typename MatType,typename EigenEquivalentType>
  EIGENPY_DEPRECATED void enableEigenPySpecific();

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

  template<typename Scalar>
  EIGEN_DONT_INLINE void exposeType()
  {
    exposeType<Scalar,0>();
  }
    
} // namespace eigenpy

#include "eigenpy/details.hpp"

#endif // ifndef __eigenpy_eigenpy_hpp__
