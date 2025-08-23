/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decomposition_sparse_cholmod_cholmod_decomposition_hpp__
#define __eigenpy_decomposition_sparse_cholmod_cholmod_decomposition_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/cholmod/CholmodBase.hpp"

namespace eigenpy {

template <typename CholdmodDerived>
struct CholmodDecompositionVisitor
    : public boost::python::def_visitor<
          CholmodDecompositionVisitor<CholdmodDerived>> {
  typedef CholdmodDerived Solver;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl

        .def(CholmodBaseVisitor<Solver>())
        .def("setMode", &Solver::setMode, bp::args("self", "mode"),
             "Set the mode for the Cholesky decomposition.");
  }
};

}  // namespace eigenpy

#endif  // ifndef
        // __eigenpy_decomposition_sparse_cholmod_cholmod_decomposition_hpp__
