/*
 * Copyright 2014-2017, Nicolas Mansard and Justin Carpentier, LAAS-CNRS
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

#ifndef __eigenpy_eigenpy_hpp__
#define __eigenpy_eigenpy_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/memory.hpp"
#include "eigenpy/deprecated.hh"

namespace eigenpy
{
  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance. */
  void enableEigenPy();

  
  template<typename MatType>
  void enableEigenPySpecific();
  
  /* Enable the Eigen--Numpy serialization for the templated MatrixBase class.
   * The second template argument is used for inheritance of Eigen classes. If
   * using a native Eigen::MatrixBase, simply repeat the same arg twice. */
  template<typename MatType,typename EigenEquivalentType>
  EIGENPY_DEPRECATED void enableEigenPySpecific();


} // namespace eigenpy

#include "eigenpy/details.hpp"

#endif // ifndef __eigenpy_eigenpy_hpp__

