/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
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
