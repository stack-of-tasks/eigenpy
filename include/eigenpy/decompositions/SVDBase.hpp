/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_svdbase_hpp__
#define __eigenpy_decompositions_svdbase_hpp__

#include <Eigen/SVD>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"
#include "eigenpy/eigen/EigenBase.hpp"

namespace eigenpy {

template <typename Derived>
struct SVDBaseVisitor
    : public boost::python::def_visitor<SVDBaseVisitor<Derived>> {
  typedef Derived Solver;

  typedef typename Derived::MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      MatrixXs;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))

        .def("computeU", &Solver::computeU, bp::arg("self"),
             "Returns true if U (full or thin) is asked for in this "
             "SVD decomposition ")
        .def("computeV", &Solver::computeV, bp::arg("self"),
             "Returns true if V (full or thin) is asked for in this "
             "SVD decomposition ")

        .def("info", &Solver::info, bp::arg("self"),
             "Reports whether previous computation was successful. ")

        .def("matrixU", &matrixU, bp::arg("self"), "Returns the matrix U.")
        .def("matrixV", &matrixV, bp::arg("self"), "Returns the matrix V.")

        .def("nonzeroSingularValues", &Solver::nonzeroSingularValues,
             bp::arg("self"),
             "Returns the number of singular values that are not exactly 0 ")
        .def("rank", &Solver::rank, bp::arg("self"),
             "the rank of the matrix of which *this is the SVD. ")

        .def("setThreshold",
             (Solver & (Solver::*)(const RealScalar &)) & Solver::setThreshold,
             bp::args("self", "threshold"),
             "Allows to prescribe a threshold to be used by certain methods, "
             "such as "
             "rank() and solve(), which need to determine when singular values "
             "are "
             "to be considered nonzero. This is not used for the SVD "
             "decomposition "
             "itself.\n"
             "\n"
             "When it needs to get the threshold value, Eigen calls "
             "threshold(). "
             "The default is NumTraits<Scalar>::epsilon()",
             bp::return_self<>())
        .def(
            "setThreshold",
            +[](Solver &self) -> Solver & {
              return self.setThreshold(Eigen::Default);
            },
            bp::arg("self"),
            "Allows to come back to the default behavior, letting Eigen use "
            "its default formula for determining the threshold.",
            bp::return_self<>())

        .def("singularValues", &Solver::singularValues, bp::arg("self"),
             "Returns the vector of singular values.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("solve", &solve<VectorXs>, bp::args("self", "b"),
             "Returns the solution x of A x = b using the current "
             "decomposition of A.")
        .def("solve", &solve<MatrixXs>, bp::args("self", "B"),
             "Returns the solution X of A X = B using the current "
             "decomposition of A where B is a right hand side matrix.")

        .def("threshold", &Solver::threshold, bp::arg("self"),
             "Returns the threshold that will be used by certain methods such "
             "as rank(). ");
  }

 private:
  static MatrixXs matrixU(const Solver &self) { return self.matrixU(); }
  static MatrixXs matrixV(const Solver &self) { return self.matrixV(); }

  template <typename MatrixOrVector>
  static MatrixOrVector solve(const Solver &self, const MatrixOrVector &vec) {
    return self.solve(vec);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_svdbase_hpp__
