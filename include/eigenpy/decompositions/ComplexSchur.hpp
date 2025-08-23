/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_complex_schur_hpp__
#define __eigenpy_decompositions_complex_schur_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct ComplexSchurVisitor
    : public boost::python::def_visitor<ComplexSchurVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::ComplexSchur<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<Eigen::DenseIndex>(bp::arg("size"), "Default constructor"))
        .def(bp::init<MatrixType, bp::optional<bool>>(
            bp::args("matrix", "computeU"), "Computes Schur  of given matrix"))

        .def("matrixU", &Solver::matrixU, bp::arg("self"),
             "Returns the unitary matrix in the Schur decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixT", &Solver::matrixT, bp::arg("self"),
             "Returns the triangular matrix in the Schur decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("compute", &ComplexSchurVisitor::compute_proxy<MatrixType>,
             bp::args("self", "matrix"), "Computes the Schur of given matrix.",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix, bool)) &
                 Solver::compute,
             bp::args("self", "matrix", "computeU"),
             "Computes the Schur of given matrix.", bp::return_self<>())

        .def("computeFromHessenberg",
             (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType>& matrixH,
                                   const Eigen::EigenBase<MatrixType>& matrixQ,
                                   bool)) &
                 Solver::computeFromHessenberg,
             bp::args("self", "matrixH", "matrixQ", "computeU"),
             "Compute Schur decomposition from a given Hessenberg matrix. ",
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
        "ComplexSchur" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(ComplexSchurVisitor())
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

#endif  // ifndef __eigenpy_decompositions_complex_schur_hpp__
