/*
 * Copyright 2016, Justin Carpentier, LAAS-CNRS
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

#ifndef __eigenpy_registration_hpp__
#define __eigenpy_registration_hpp__

#include <boost/python.hpp>

namespace eigenpy
{
  ///
  /// \brief Check at runtime the registration of the type T inside the boost python registry.
  ///
  /// \tparam T The type to check the registration.
  ///
  /// \returns true if the type T is already registered.
  ///
  template<typename T>
  inline bool check_registration()
  {
    namespace bp = boost::python;
    
    const bp::type_info info = bp::type_id<T>();
    const bp::converter::registration* reg = bp::converter::registry::query(info);
    if (reg == NULL) return false;
    else if ((*reg).m_to_python == NULL) return false;
    
    return true;
  }
}

#endif // ifndef __eigenpy_registration_hpp__
