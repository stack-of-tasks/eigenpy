/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_complex_eigen_solver_hpp__
#define __eigenpy_decompositions_complex_eigen_solver_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct ComplexEigenSolverVisitor : public boost::python::def_visitor<
                                       ComplexEigenSolverVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::ComplexEigenSolver<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>("Default constructor"))
        .def(bp::init<Eigen::DenseIndex>(
            bp::arg("size"), "Default constructor with memory preallocation"))
        .def(bp::init<MatrixType, bp::optional<bool>>(
            bp::args("matrix", "compute_eigen_vectors"),
            "Computes eigendecomposition of given matrix"))

        .def("eigenvalues", &Solver::eigenvalues, bp::arg("self"),
             "Returns the eigenvalues of given matrix.",
             bp::return_internal_reference<>())
        .def("eigenvectors", &Solver::eigenvectors, bp::arg("self"),
             "Returns the eigenvectors of given matrix.",
             bp::return_internal_reference<>())

        .def("compute", &ComplexEigenSolverVisitor::compute_proxy<MatrixType>,
             bp::args("self", "matrix"),
             "Computes the eigendecomposition of given matrix.",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix, bool)) &
                 Solver::compute,
             bp::args("self", "matrix", "compute_eigen_vectors"),
             "Computes the eigendecomposition of given matrix.",
             bp::return_self<>())

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("getMaxIterations", &Solver::getMaxIterations, bp::arg("self"),
             "Returns the maximum number of iterations.")
        .def("setMaxIterations", &Solver::setMaxIterations,
             bp::args("self", "max_iter"),
             "Sets the maximum number of iterations allowed.",
             bp::return_self<>());
  }

  static void expose() {
    static const std::string classname =
        "ComplexEigenSolver" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(ComplexEigenSolverVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixType>
  static Solver& compute_proxy(Solver& self,
                               const Eigen::EigenBase<MatrixType>& matrix) {
    return self.compute(matrix);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_complex_eigen_solver_hpp__
