/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_generalized_eigen_solver_hpp__
#define __eigenpy_decompositions_generalized_eigen_solver_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct GeneralizedEigenSolverVisitor
    : public boost::python::def_visitor<
          GeneralizedEigenSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::GeneralizedEigenSolver<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>("Default constructor"))
        .def(bp::init<Eigen::DenseIndex>(
            bp::arg("size"), "Default constructor with memory preallocation. "))
        .def(bp::init<MatrixType, MatrixType, bp::optional<bool>>(
            bp::args("A", "B", "computeEigenVectors"),
            "Computes the generalized eigendecomposition of given matrix "
            "pair. "))

        .def("eigenvectors", &Solver::eigenvectors, bp::arg("self"),
             "Returns an expression of the computed generalized eigenvectors. ")
        .def(
            "eigenvalues",
            +[](const Solver& c) { return c.eigenvalues().eval(); },
            "Returns the computed generalized eigenvalues.")

        .def("alphas", &Solver::alphas, bp::arg("self"),
             "Returns the vectors containing the alpha values. ")
        .def("betas", &Solver::betas, bp::arg("self"),
             "Returns the vectors containing the beta values. ")

        .def("compute",
             &GeneralizedEigenSolverVisitor::compute_proxy<MatrixType>,
             bp::args("self", "A", "B"),
             "Computes generalized eigendecomposition of given matrix. ",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const MatrixType& A, const MatrixType& B, bool)) &
                 Solver::compute,
             bp::args("self", "A", "B", "computeEigenvectors"),
             "Computes generalized eigendecomposition of given matrix. .",
             bp::return_self<>())

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("setMaxIterations", &Solver::setMaxIterations,
             bp::args("self", "max_iter"),
             "Sets the maximum number of iterations allowed.",
             bp::return_self<>());
  }

  static void expose() {
    static const std::string classname =
        "GeneralizedEigenSolver" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(GeneralizedEigenSolverVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixType>
  static Solver& compute_proxy(Solver& self, const MatrixType& A,
                               const MatrixType& B) {
    return self.compute(A, B);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_generalized_eigen_solver_hpp__
