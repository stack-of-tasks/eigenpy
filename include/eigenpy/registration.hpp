/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_registration_hpp__
#define __eigenpy_registration_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {

///
/// \brief Check at runtime the registration of the type T inside the boost
/// python registry.
///
/// \tparam T The type to check the registration.
///
/// \returns true if the type T is already registered.
///
template <typename T>
inline bool check_registration() {
  const bp::type_info info = bp::type_id<T>();
  const bp::converter::registration* reg = bp::converter::registry::query(info);
  if (reg == NULL)
    return false;
  else if ((*reg).m_to_python == NULL)
    return false;

  return true;
}

///
/// \brief Symlink to the current scope the already registered class T.
///
///  \returns true if the type T is effectively symlinked.
///
/// \tparam T The type to symlink.
///
template <typename T>
inline bool register_symbolic_link_to_registered_type() {
  if (eigenpy::check_registration<T>()) {
    const bp::type_info info = bp::type_id<T>();
    const bp::converter::registration* reg =
        bp::converter::registry::query(info);
    bp::handle<> class_obj(reg->get_class_object());
    bp::scope().attr(reg->get_class_object()->tp_name) = bp::object(class_obj);
    return true;
  }

  return false;
}
}  // namespace eigenpy

#endif  // ifndef __eigenpy_registration_hpp__
