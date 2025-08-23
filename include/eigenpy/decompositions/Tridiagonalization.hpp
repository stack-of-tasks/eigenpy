/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_tridiagonalization_hpp__
#define __eigenpy_decompositions_tridiagonalization_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct TridiagonalizationVisitor : public boost::python::def_visitor<
                                       TridiagonalizationVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Tridiagonalization<MatrixType> Solver;
  typedef Eigen::VectorXd VectorType;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(
          bp::init<Eigen::DenseIndex>(bp::arg("size"), "Default constructor. "))
        .def(bp::init<MatrixType>(bp::arg("matrix"),
                                  "Constructor; computes tridiagonal "
                                  "decomposition of given matrix. "))

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> &matrix)) &
                Solver::compute,
            bp::args("self", "matrix"),
            "Computes tridiagonal decomposition of given matrix. ",
            bp::return_self<>())

        .def("householderCoefficients", &Solver::householderCoefficients,
             bp::arg("self"), "Returns the Householder coefficients. ")
        .def("packedMatrix", &Solver::packedMatrix, bp::arg("self"),
             "Returns the internal representation of the decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())

        .def(
            "matrixQ",
            +[](const Solver &c) -> MatrixType { return c.matrixQ(); },
            "Returns the unitary matrix Q in the decomposition.")
        .def(
            "matrixT",
            +[](const Solver &c) -> MatrixType { return c.matrixT(); },
            "Returns an expression of the tridiagonal matrix T in the "
            "decomposition.")

        .def(
            "diagonal",
            +[](const Solver &c) -> VectorType { return c.diagonal(); },
            bp::arg("self"),
            "Returns the diagonal of the tridiagonal matrix T in the "
            "decomposition. ")

        .def(
            "subDiagonal",
            +[](const Solver &c) -> VectorType { return c.subDiagonal(); },
            bp::arg("self"),
            "Returns the subdiagonal of the tridiagonal matrix T in the "
            "decomposition.");
  }

  static void expose() {
    static const std::string classname =
        "TridiagonalizationVisitor" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(TridiagonalizationVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_tridiagonalization_hpp__
