/*
 * Copyright 2017 CNRS
 * Copyright 2024-2025 INRIA
 */

#ifndef __eigenpy_solvers_bfgs_preconditioners_hpp__
#define __eigenpy_solvers_bfgs_preconditioners_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/BasicPreconditioners.hpp"

namespace eigenpy {

template <typename Preconditioner>
struct BFGSPreconditionerBaseVisitor
    : public bp::def_visitor<BFGSPreconditionerBaseVisitor<Preconditioner>> {
  typedef Eigen::VectorXd VectorType;
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def("rows", &Preconditioner::rows,
             "Returns the number of rows in the preconditioner.")
        .def("cols", &Preconditioner::cols,
             "Returns the number of cols in the preconditioner.")
        .def("dim", &Preconditioner::dim,
             "Returns the dimension of the BFGS preconditioner")
        .def("update",
             (const Preconditioner& (Preconditioner::*)(const VectorType&,
                                                        const VectorType&)
                  const) &
                 Preconditioner::update,
             bp::args("s", "y"), "Update the BFGS estimate of the matrix A.",
             bp::return_value_policy<bp::reference_existing_object>())
        .def("reset", &Preconditioner::reset, "Reset the BFGS estimate.");
  }

  static void expose(const std::string& name) {
    bp::class_<Preconditioner>(name, bp::no_init)
        .def(IdVisitor<Preconditioner>())
        .def(BFGSPreconditionerBaseVisitor<Preconditioner>());
  }
};

template <typename Preconditioner>
struct LimitedBFGSPreconditionerBaseVisitor
    : public bp::def_visitor<
          LimitedBFGSPreconditionerBaseVisitor<Preconditioner>> {
  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(PreconditionerBaseVisitor<Preconditioner>())
        .def(BFGSPreconditionerBaseVisitor<Preconditioner>())
        .def("resize", &Preconditioner::resize, bp::arg("dim"),
             "Resizes the preconditionner with size dim.",
             bp::return_value_policy<bp::reference_existing_object>());
  }

  static void expose(const std::string& name) {
    bp::class_<Preconditioner>(name.c_str(), bp::no_init)
        .def(IdVisitor<Preconditioner>())
        .def(LimitedBFGSPreconditionerBaseVisitor<Preconditioner>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_solvers_bfgs_preconditioners_hpp__
