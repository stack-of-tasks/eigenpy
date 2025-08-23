/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_hessenberg_decomposition_hpp__
#define __eigenpy_decompositions_hessenberg_decomposition_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct HessenbergDecompositionVisitor
    : public boost::python::def_visitor<
          HessenbergDecompositionVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::HessenbergDecomposition<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl
        .def(bp::init<Eigen::DenseIndex>(
            bp::arg("size"),
            "Default constructor; the decomposition will be computed later. "))
        .def(bp::init<MatrixType>(
            bp::arg("matrix"),
            "Constructor; computes Hessenberg decomposition of given matrix. "))

        .def(
            "compute",
            (Solver & (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix)) &
                Solver::compute,
            bp::args("self", "A"),
            "Computes Hessenberg decomposition of given matrix. ",
            bp::return_self<>())

        .def("householderCoefficients", &Solver::householderCoefficients,
             bp::arg("self"), "Returns the Householder coefficients. ",
             bp::return_value_policy<bp::copy_const_reference>())

        .def(
            "matrixQ",
            +[](const Solver& c) -> MatrixType { return c.matrixQ(); },
            "Reconstructs the orthogonal matrix Q in the decomposition.")
        .def(
            "matrixH",
            +[](const Solver& c) -> MatrixType { return c.matrixH(); },
            "Constructs the Hessenberg matrix H in the decomposition.")

        .def("packedMatrix", &Solver::packedMatrix, bp::arg("self"),
             "Returns the internal representation of the decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>());
  }

  static void expose() {
    static const std::string classname =
        "HessenbergDecomposition" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(HessenbergDecompositionVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_hessenberg_decomposition_hpp__
