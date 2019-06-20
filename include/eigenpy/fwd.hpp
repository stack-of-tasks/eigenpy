/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_fwd_hpp__
#define __eigenpy_fwd_hpp__

#include <boost/python.hpp>
#include <Eigen/Core>

#include <numpy/numpyconfig.h>
#ifdef NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/noprefix.h>

#ifdef NPY_ALIGNED
#if EIGEN_VERSION_AT_LEAST(3,2,90)
  #define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Aligned16
#else
  #define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Aligned
#endif
#else
  #define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Unaligned
#endif

#include "eigenpy/expose.hpp"

#endif // ifndef __eigenpy_fwd_hpp__

