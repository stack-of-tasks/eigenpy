//
// Copyright (c) 2020-2023 INRIA
//

#ifndef __eigenpy_eigen_typedef_hpp__
#define __eigenpy_eigen_typedef_hpp__

#include "eigenpy/fwd.hpp"

#define EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, Size, SizeSuffix) \
  /** \ingroup matrixtypedefs */                                           \
  typedef Eigen::Matrix<Type, Size, Size, Options>                         \
      Matrix##SizeSuffix##TypeSuffix;                                      \
  /** \ingroup matrixtypedefs */                                           \
  typedef Eigen::Matrix<Type, Size, 1> Vector##SizeSuffix##TypeSuffix;     \
  /** \ingroup matrixtypedefs */                                           \
  typedef Eigen::Matrix<Type, 1, Size> RowVector##SizeSuffix##TypeSuffix;

#define EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, Size) \
  /** \ingroup matrixtypedefs */                                     \
  typedef Eigen::Matrix<Type, Size, Eigen::Dynamic, Options>         \
      Matrix##Size##X##TypeSuffix;                                   \
  /** \ingroup matrixtypedefs */                                     \
  typedef Eigen::Matrix<Type, Eigen::Dynamic, Size, Options>         \
      Matrix##X##Size##TypeSuffix;

#define EIGENPY_MAKE_TYPEDEFS_ALL_SIZES(Type, Options, TypeSuffix)    \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 2, 2)              \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 3, 3)              \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 4, 4)              \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, Eigen::Dynamic, X) \
  EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, 2)           \
  EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, 3)           \
  EIGENPY_MAKE_FIXED_TYPEDEFS(Type, Options, TypeSuffix, 4)           \
  EIGENPY_MAKE_TYPEDEFS(Type, Options, TypeSuffix, 1, 1)

#endif  // ifndef __eigenpy_eigen_typedef_hpp__
