/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2024, INRIA
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

template <typename Scalar>
void exposeIsApprox() {
  enum { Options = 0 };
  EIGENPY_MAKE_TYPEDEFS(Scalar, Options, s, Eigen::Dynamic, X);
  EIGENPY_UNUSED_TYPE(VectorXs);
  EIGENPY_UNUSED_TYPE(RowVectorXs);
  //  typedef Eigen::SparseMatrix<Scalar, Options> SparseMatrixXs;
  typedef typename MatrixXs::RealScalar RealScalar;

  using namespace Eigen;
  const RealScalar dummy_precision =
      Eigen::NumTraits<RealScalar>::dummy_precision();

  bp::def("is_approx",
          (bool (*)(const Eigen::MatrixBase<MatrixXs> &,
                    const Eigen::MatrixBase<MatrixXs> &,
                    const RealScalar &))&is_approx,
          (bp::arg("A"), bp::arg("B"), bp::arg("prec") = dummy_precision),
          "Returns True if A is approximately equal to B, within the "
          "precision determined by prec.");

  //  bp::def("is_approx",
  //          (bool (*)(const Eigen::SparseMatrixBase<SparseMatrixXs> &,
  //                    const Eigen::SparseMatrixBase<SparseMatrixXs> &,
  //                    const RealScalar &)) &
  //              is_approx,
  //          (bp::arg("A"), bp::arg("B"), bp::arg("prec") = dummy_precision),
  //          "Returns True if A is approximately equal to B, within the "
  //          "precision determined by prec.");
}

BOOST_PYTHON_MODULE(eigenpy_pywrap) {
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

  exposeIsApprox<double>();
  exposeIsApprox<std::complex<double>>();

  exposeDecompositions();
}
