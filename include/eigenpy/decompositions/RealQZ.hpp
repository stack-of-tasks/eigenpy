/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_generalized_real_qz_hpp__
#define __eigenpy_decompositions_generalized_real_qz_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct RealQZVisitor
    : public boost::python::def_visitor<RealQZVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::RealQZ<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(
          bp::init<Eigen::DenseIndex>(bp::arg("size"), "Default constructor. "))
        .def(bp::init<MatrixType, MatrixType, bp::optional<bool>>(
            bp::args("A", "B", "computeQZ"),
            "Constructor; computes real QZ decomposition of given matrices.  "))

        .def("compute", &RealQZVisitor::compute_proxy<MatrixType>,
             bp::args("self", "A", "B"),
             "Computes QZ decomposition of given matrix.  ",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const MatrixType& A, const MatrixType& B, bool)) &
                 Solver::compute,
             bp::args("self", "A", "B", "computeEigenvectors"),
             "Computes QZ decomposition of given matrix. ", bp::return_self<>())

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")

        .def("matrixQ", &Solver::matrixQ, bp::arg("self"),
             "Returns matrix Q in the QZ decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixS", &Solver::matrixS, bp::arg("self"),
             "Returns matrix S in the QZ decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixT", &Solver::matrixT, bp::arg("self"),
             "Returns matrix T in the QZ decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())
        .def("matrixZ", &Solver::matrixZ, bp::arg("self"),
             "Returns matrix Z in the QZ decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("iterations", &Solver::iterations, bp::arg("self"),
             "Returns number of performed QR-like iterations. ")
        .def("setMaxIterations", &Solver::setMaxIterations,
             bp::args("self", "max_iter"),
             "Sets the maximum number of iterations allowed.",
             bp::return_self<>());
  }

  static void expose() {
    static const std::string classname =
        "RealQZVisitor" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(RealQZVisitor())
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

#endif  // ifndef __eigenpy_decompositions_generalized_real_qz_hpp__
