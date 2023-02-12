/*
 * Copyright 2017, Justin Carpentier, LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __eigenpy_bfgs_preconditioners_hpp__
#define __eigenpy_bfgs_preconditioners_hpp__

#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/fwd.hpp"
#include "eigenpy/solvers/BasicPreconditioners.hpp"

namespace eigenpy {

template <typename Preconditioner>
struct BFGSPreconditionerBaseVisitor
    : public bp::def_visitor<BFGSPreconditionerBaseVisitor<Preconditioner> > {
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
        .def(BFGSPreconditionerBaseVisitor<Preconditioner>());
  }
};

template <typename Preconditioner>
struct LimitedBFGSPreconditionerBaseVisitor
    : public bp::def_visitor<
          LimitedBFGSPreconditionerBaseVisitor<Preconditioner> > {
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
        .def(LimitedBFGSPreconditionerBaseVisitor<Preconditioner>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_bfgs_preconditioners_hpp__
