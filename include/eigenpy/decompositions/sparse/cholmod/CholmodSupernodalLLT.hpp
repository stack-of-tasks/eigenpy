/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decomposition_sparse_cholmod_cholmod_supernodal_llt_hpp__
#define __eigenpy_decomposition_sparse_cholmod_cholmod_supernodal_llt_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/cholmod/CholmodDecomposition.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename MatrixType_, int UpLo_ = Eigen::Lower>
struct CholmodSupernodalLLTVisitor
    : public boost::python::def_visitor<
          CholmodSupernodalLLTVisitor<MatrixType_, UpLo_>> {
  typedef MatrixType_ MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  typedef Eigen::CholmodSupernodalLLT<MatrixType_, UpLo_> Solver;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl

        .def(CholmodBaseVisitor<Solver>())
        .def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the LLT "
                                  "factorization from a given matrix."))

        ;
  }

  static void expose() {
    static const std::string classname =
        "CholmodSupernodalLLT_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "A supernodal direct Cholesky (LLT) factorization and solver based on "
        "Cholmod.\n\n"
        "This class allows to solve for A.X = B sparse linear problems via a "
        "supernodal LL^T Cholesky factorization using the Cholmod library."
        "This supernodal variant performs best on dense enough problems, e.g., "
        "3D FEM, or very high order 2D FEM."
        "The sparse matrix A must be selfadjoint and positive definite. The "
        "vectors or matrices X and B can be either dense or sparse.",
        bp::no_init)
        .def(CholmodSupernodalLLTVisitor());
  }
};

}  // namespace eigenpy

#endif  // ifndef
        // __eigenpy_decomposition_sparse_cholmod_cholmod_supernodal_llt_hpp__
