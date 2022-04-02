/*
 * Copyright 2019 INRIA
 */

#ifndef __eigenpy_expose_hpp__
#define __eigenpy_expose_hpp__

#include "eigenpy/registration.hpp"

namespace eigenpy {
///
/// \brief Allows a template specialization.
///
template <typename T>
struct call {
  static inline void expose() { T::expose(); }
};

///
/// \brief Call the expose function of a given type T.
///
template <typename T>
inline void expose() {
  if (!register_symbolic_link_to_registered_type<T>()) call<T>::expose();
}
}  // namespace eigenpy

#endif  // ifndef __eigenpy_expose_hpp__
