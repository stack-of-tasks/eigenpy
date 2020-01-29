/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/version.hpp"
#include "eigenpy/geometry.hpp"

#include "eigenpy/computation-info.hpp"

#include "eigenpy/solvers/solvers.hpp"
#include "eigenpy/solvers/preconditioners.hpp"
#include "eigenpy/decompositions/decompositions.hpp"

#include "eigenpy/utils/is-approx.hpp"

#include <boost/python/scope.hpp>

#define DEFINE_IS_APPROX(MatType) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(is_approx_overload##MatType,eigenpy::is_approx,2,3)

#define EXPOSE_IS_APPROX(MatType) \
  bp::def("is_approx", \
          (bool (*)(const Eigen::MatrixBase<MatType> &, \
                    const Eigen::MatrixBase<MatType> &, \
                    const MatType::Scalar &))eigenpy::is_approx<MatType,MatType>, \
                    is_approx_overload##MatType(bp::args("A","B","prec"), \
          "Returns True if A is approximately equal to B, within the precision determined by prec."))


using namespace eigenpy;

DEFINE_IS_APPROX(MatrixXd)
DEFINE_IS_APPROX(MatrixXf)


BOOST_PYTHON_MODULE(eigenpy)
{
  namespace bp = boost::python;
  enableEigenPy();
  
  bp::scope().attr("__version__") = eigenpy::printVersion();
  bp::scope().attr("__raw_version__") = bp::str(EIGENPY_VERSION);
  bp::def("checkVersionAtLeast",&eigenpy::checkVersionAtLeast,
          bp::args("major_version","minor_version","patch_version"),
          "Checks if the current version of EigenPy is at least the version provided by the input arguments.");
  
  exposeAngleAxis();
  exposeQuaternion();
  exposeGeometryConversion();
  
  exposeComputationInfo();
  
  {
    bp::scope solvers = boost::python::class_<SolversScope>("solvers");
    exposeSolvers();
    exposePreconditioners();
    
    register_symbolic_link_to_registered_type<Eigen::ComputationInfo>();
  }
  
  {
    using namespace Eigen;
    EXPOSE_IS_APPROX(MatrixXd);
    EXPOSE_IS_APPROX(MatrixXf);
  }
  
  exposeDecompositions();
}
