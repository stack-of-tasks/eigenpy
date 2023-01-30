/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2023, INRIA
 */

#include <boost/python/scope.hpp>

#include "eigenpy/computation-info.hpp"
#include "eigenpy/decompositions/decompositions.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/geometry.hpp"
#include "eigenpy/solvers/preconditioners.hpp"
#include "eigenpy/solvers/solvers.hpp"
#include "eigenpy/std-vector.hpp"
#include "eigenpy/utils/is-approx.hpp"
#include "eigenpy/version.hpp"

using namespace eigenpy;

BOOST_PYTHON_MODULE(eigenpy_pywrap) {
  namespace bp = boost::python;
  enableEigenPy();

  bp::scope().attr("__version__") = eigenpy::printVersion();
  bp::scope().attr("__eigen_version__") = eigenpy::printEigenVersion();
  bp::scope().attr("__raw_version__") = bp::str(EIGENPY_VERSION);
  bp::def("checkVersionAtLeast", &eigenpy::checkVersionAtLeast,
          bp::args("major_version", "minor_version", "patch_version"),
          "Checks if the current version of EigenPy is at least the version "
          "provided by the input arguments.");

  bp::def("SimdInstructionSetsInUse", &Eigen::SimdInstructionSetsInUse,
          "Get the set of SIMD instructions in use with Eigen.");

  exposeAngleAxis();
  exposeQuaternion();
  exposeGeometryConversion();
  exposeStdVector();

  exposeComputationInfo();

  {
    bp::scope solvers = boost::python::class_<SolversScope>("solvers");
    exposeSolvers();
    exposePreconditioners();

    register_symbolic_link_to_registered_type<Eigen::ComputationInfo>();
  }

  {
    using namespace Eigen;

    bp::def("is_approx",
            (bool (*)(const Eigen::MatrixBase<MatrixXd> &,
                      const Eigen::MatrixBase<MatrixXd> &, const double &)) &
                is_approx<MatrixXd, MatrixXd>,
            (bp::arg("A"), bp::arg("B"), bp::arg("prec") = 1e-12),
            "Returns True if A is approximately equal to B, within the "
            "precision determined by prec.");
  }

  exposeDecompositions();
}
