/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decomposition_sparse_cholmod_cholmod_simplicial_ldlt_hpp__
#define __eigenpy_decomposition_sparse_cholmod_cholmod_simplicial_ldlt_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/cholmod/CholmodDecomposition.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename MatrixType_, int UpLo_ = Eigen::Lower>
struct CholmodSimplicialLDLTVisitor
    : public boost::python::def_visitor<
          CholmodSimplicialLDLTVisitor<MatrixType_, UpLo_>> {
  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  typedef Eigen::CholmodSimplicialLDLT<MatrixType_, UpLo_> Solver;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl

        .def(CholmodBaseVisitor<Solver>())
        .def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the LDLT "
                                  "factorization from a given matrix."))

        ;
  }

  static void expose() {
    static const std::string classname =
        "CholmodSimplicialLDLT_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "A simplicial direct Cholesky (LDLT) factorization and solver based on "
        "Cholmod.\n\n"
        "This class allows to solve for A.X = B sparse linear problems via a "
        "simplicial LL^T Cholesky factorization using the Cholmod library."
        "This simplicial variant is equivalent to Eigen's built-in "
        "SimplicialLDLT class."
        "Therefore, it has little practical interest. The sparse matrix A must "
        "be selfadjoint and positive definite."
        "The vectors or matrices X and B can be either dense or sparse.",
        bp::no_init)
        .def(CholmodSimplicialLDLTVisitor());
  }
};

}  // namespace eigenpy

#endif  // ifndef
        // __eigenpy_decomposition_sparse_cholmod_cholmod_simplicial_ldlt_hpp__
