/*
 * Copyright 2014-2020 CNRS INRIA
 */

#ifndef __eigenpy_fwd_hpp__
#define __eigenpy_fwd_hpp__

#include "eigenpy/config.hpp"

#include <boost/python.hpp>
#include <Eigen/Core>

#define NO_IMPORT_ARRAY
  #include "eigenpy/numpy.hpp"
#undef NO_IMPORT_ARRAY

#if EIGEN_VERSION_AT_LEAST(3,2,90)
  #define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Aligned16
#else
  #define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Aligned
#endif

#define EIGENPY_NO_ALIGNMENT_VALUE Eigen::Unaligned

#define EIGENPY_UNUSED_VARIABLE(var) (void)(var)

#include "eigenpy/expose.hpp"

namespace eigenpy
{
  template<typename MatType> struct EigenToPy;
  template<typename MatType> struct EigenFromPy;
}

#endif // ifndef __eigenpy_fwd_hpp__
