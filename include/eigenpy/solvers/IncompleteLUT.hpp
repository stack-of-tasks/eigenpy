/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_solvers_incomplete_lut_hpp__
#define __eigenpy_solvers_incomplete_lut_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct IncompleteLUTVisitor
    : public boost::python::def_visitor<IncompleteLUTVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::IncompleteLUT<Scalar> Solver;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      DenseVectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      DenseMatrixXs;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the LDLT "
                                  "factorization from a given matrix."))
        .def(bp::init<const MatrixType&, RealScalar, int>(
            (bp::arg("matrix"),
             bp::arg("droptol") = Eigen::NumTraits<Scalar>::dummy_precision(),
             bp::arg("fillfactor") = 10),
            "Constructs an incomplete LU factorization from a given matrix."))

        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows of the matrix.")
        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of cols of the matrix.")

        .def("info", &Solver::info, bp::arg("self"),
             "Reports whether previous computation was successful.")

        .def(
            "analyzePattern",
            +[](Solver& self, const MatrixType& amat) {
              self.analyzePattern(amat);
            },
            bp::arg("matrix"))
        .def(
            "factorize",
            +[](Solver& self, const MatrixType& amat) { self.factorize(amat); },
            bp::arg("matrix"))
        .def(
            "compute",
            +[](Solver& self, const MatrixType& amat) { self.compute(amat); },
            bp::arg("matrix"))

        .def("setDroptol", &Solver::setDroptol, bp::arg("self"))
        .def("setFillfactor", &Solver::setFillfactor, bp::arg("self"))

        .def(
            "solve",
            +[](const Solver& self, const Eigen::Ref<DenseVectorXs const>& b)
                -> DenseVectorXs { return self.solve(b); },
            bp::arg("b"),
            "Returns the solution x of A x = b using the current decomposition "
            "of A, where b is a right hand side vector.")
        .def(
            "solve",
            +[](const Solver& self, const Eigen::Ref<DenseMatrixXs const>& B)
                -> DenseMatrixXs { return self.solve(B); },
            bp::arg("b"),
            "Returns the solution X of A X = B using the current decomposition "
            "of A where B is a right hand side matrix.")
        .def(
            "solve",
            +[](const Solver& self, const MatrixType& B) -> MatrixType {
              DenseMatrixXs B_dense = DenseMatrixXs(B);
              DenseMatrixXs X_dense = self.solve(B_dense);
              return MatrixType(X_dense.sparseView());
            },
            bp::arg("b"),
            "Returns the solution X of A X = B using the current decomposition "
            "of A where B is a right hand side matrix.");
  }

  static void expose() {
    static const std::string classname =
        "IncompleteLUT_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "Incomplete LU factorization with dual-threshold strategy.",
        bp::no_init)
        .def(IncompleteLUTVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_incomplete_lut_hpp__
