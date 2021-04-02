/*
 * Copyright 2014-2020 CNRS INRIA
 */

#ifndef __eigenpy_fwd_hpp__
#define __eigenpy_fwd_hpp__

#include "eigenpy/config.hpp"

// Silence a warning about a deprecated use of boost bind by boost python
// at least fo boost 1.73 to 1.75
// ref. https://github.com/stack-of-tasks/tsid/issues/128
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <boost/python/scope.hpp>

#define NO_IMPORT_ARRAY
  #include "eigenpy/numpy.hpp"
#undef NO_IMPORT_ARRAY

#undef BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <Eigen/Core>

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
  template<typename MatType, typename Scalar = typename boost::remove_reference<MatType>::type::Scalar> struct EigenToPy;
  template<typename MatType, typename Scalar = typename boost::remove_reference<MatType>::type::Scalar> struct EigenFromPy;
}

#endif // ifndef __eigenpy_fwd_hpp__
