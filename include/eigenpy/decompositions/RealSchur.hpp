/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_generalized_real_schur_hpp__
#define __eigenpy_decompositions_generalized_real_schur_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct RealSchurVisitor
    : public boost::python::def_visitor<RealSchurVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::RealSchur<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(
          bp::init<Eigen::DenseIndex>(bp::arg("size"), "Default constructor. "))
        .def(bp::init<MatrixType, bp::optional<bool>>(
            bp::args("matrix", "computeU"),
            "Constructor; computes real Schur decomposition of given matrix. "))

        .def("compute", &RealSchurVisitor::compute_proxy<MatrixType>,
             bp::args("self", "matrix"),
             "Computes Schur decomposition of given matrix. ",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix, bool)) &
                 Solver::compute,
             bp::args("self", "matrix", "computeEigenvectors"),
             "Computes Schur decomposition of given matrix. ",
             bp::return_self<>())

        .def("computeFromHessenberg",
             (Solver & (Solver::*)(const MatrixType& matrixH,
                                   const MatrixType& matrixQ, bool)) &
                 Solver::computeFromHessenberg,
             bp::args("self", "matrixH", "matrixQ", "computeU"),
             "Compute Schur decomposition from a given Hessenberg matrix. ",
             bp::return_self<>())

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("matrixT", &Solver::matrixT, bp::arg("self"),
             "Returns the quasi-triangular matrix in the Schur decomposition.",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixU", &Solver::matrixU, bp::arg("self"),
             "Returns the orthogonal matrix in the Schur decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("setMaxIterations", &Solver::setMaxIterations,
             bp::args("self", "max_iter"),
             "Sets the maximum number of iterations allowed.",
             bp::return_self<>())
        .def("getMaxIterations", &Solver::getMaxIterations, bp::arg("self"),
             "Returns the maximum number of iterations.");
  }

  static void expose() {
    static const std::string classname =
        "RealSchurVisitor" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(RealSchurVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixType>
  static Solver& compute_proxy(Solver& self, const MatrixType& A) {
    return self.compute(A);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_generalized_real_schur_hpp__
