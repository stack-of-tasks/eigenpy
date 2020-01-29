/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_decompositions_computation_info_hpp__
#define __eigenpy_decompositions_computation_info_hpp__

#include <Eigen/Core>
#include <boost/python.hpp>

#include "eigenpy/eigenpy_export.h"

namespace eigenpy
{
  inline void EIGENPY_EXPORT exposeComputationInfo()
  {
    boost::python::enum_<Eigen::ComputationInfo>("ComputationInfo")
    .value("Success",Eigen::Success)
    .value("NumericalIssue",Eigen::NumericalIssue)
    .value("NoConvergence",Eigen::NoConvergence)
    .value("InvalidInput",Eigen::InvalidInput)
    ;
  }
} // namespace eigenpy

#endif // define __eigenpy_decompositions_computation_info_hpp__
