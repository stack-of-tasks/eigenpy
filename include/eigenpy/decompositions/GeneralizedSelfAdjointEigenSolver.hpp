/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_generalized_self_adjoint_eigen_solver_hpp__
#define __eigenpy_decompositions_generalized_self_adjoint_eigen_solver_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"
#include "eigenpy/decompositions/SelfAdjointEigenSolver.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct GeneralizedSelfAdjointEigenSolverVisitor
    : public boost::python::def_visitor<
          GeneralizedSelfAdjointEigenSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::GeneralizedSelfAdjointEigenSolver<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>("Default constructor"))
        .def(bp::init<Eigen::DenseIndex>(
            bp::arg("size"), "Default constructor with memory preallocation. "))
        .def(bp::init<MatrixType, MatrixType, bp::optional<int>>(
            bp::args("matA", "matB", "options"),
            "Constructor: Computes generalized eigendecomposition of given "
            "matrix pencil. "))

        .def("compute",
             &GeneralizedSelfAdjointEigenSolverVisitor::compute_proxy<
                 MatrixType>,
             bp::args("self", "A", "B"),
             "Computes generalized eigendecomposition of given matrix pencil. ",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const MatrixType& A, const MatrixType& B, int)) &
                 Solver::compute,
             bp::args("self", "A", "B", "options"),
             "Computes generalized eigendecomposition of given matrix pencil.",
             bp::return_self<>());
  }

  static void expose() {
    static const std::string classname =
        "GeneralizedSelfAdjointEigenSolver" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver, bp::bases<Eigen::SelfAdjointEigenSolver<MatrixType>>>(
        name.c_str(), bp::no_init)
        .def(GeneralizedSelfAdjointEigenSolverVisitor())
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

#endif  // ifndef
        // __eigenpy_decompositions_generalized_self_adjoint_eigen_solver_hpp__
