/*
 * Copyright 2017-2018, Justin Carpentier, LAAS-CNRS
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

#include <Eigen/Core>

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)
#include "eigenpy/solvers/BasicPreconditioners.hpp"
#include "eigenpy/solvers/preconditioners.hpp"
//#include "eigenpy/solvers/BFGSPreconditioners.hpp"

namespace eigenpy {

void exposePreconditioners() {
  using namespace Eigen;

  DiagonalPreconditionerVisitor<double>::expose();
#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  LeastSquareDiagonalPreconditionerVisitor<double>::expose();
#endif
  IdentityPreconditionerVisitor::expose();
  //      LimitedBFGSPreconditionerBaseVisitor<
  //      LimitedBFGSPreconditioner<double,Eigen::Dynamic,Eigen::Upper|Eigen::Lower>
  //      >::expose("LimitedBFGSPreconditioner");
}

}  // namespace eigenpy

#endif
