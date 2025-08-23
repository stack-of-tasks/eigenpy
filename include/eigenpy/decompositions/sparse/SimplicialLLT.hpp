/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_sparse_simplicial_llt_hpp__
#define __eigenpy_decompositions_sparse_simplicial_llt_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/SimplicialCholesky.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType, int _UpLo = Eigen::Lower,
          typename _Ordering =
              Eigen::AMDOrdering<typename _MatrixType::StorageIndex>>
struct SimplicialLLTVisitor
    : public boost::python::def_visitor<
          SimplicialLLTVisitor<_MatrixType, _UpLo, _Ordering>> {
  typedef SimplicialLLTVisitor<_MatrixType, _UpLo, _Ordering> Visitor;
  typedef _MatrixType MatrixType;

  typedef Eigen::SimplicialLLT<MatrixType> Solver;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      DenseVectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      DenseMatrixXs;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(bp::args("self", "matrix"),
                                  "Constructs and performs the LLT "
                                  "factorization from a given matrix."))

        .def(SimplicialCholeskyVisitor<Solver>());
  }

  static void expose() {
    static const std::string classname =
        "SimplicialLLT_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "A direct sparse LLT Cholesky factorizations.\n\n"
        "This class provides a LL^T Cholesky factorizations of sparse matrices "
        "that are selfadjoint and positive definite."
        "The factorization allows for solving A.X = B where X and B can be "
        "either dense or sparse.\n\n"
        "In order to reduce the fill-in, a symmetric permutation P is applied "
        "prior to the factorization such that the factorized matrix is P A "
        "P^-1.",
        bp::no_init)
        .def(SimplicialLLTVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_sparse_simplicial_llt_hpp__
