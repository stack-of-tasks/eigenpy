/*
 * Copyright 2019, INRIA
 */

#ifndef __eigenpy_expose_hpp__
#define __eigenpy_expose_hpp__

#include "eigenpy/registration.hpp"

namespace eigenpy
{
  namespace internal
  {
    ///
    /// \brief Allows a template specialization.
    ///
    template<typename T>
    struct call_expose
    {
      static inline void run() { T::expose(); }
    };
  } // namespace internal
  
  ///
  /// \brief Call the expose function of a given type T.
  ///
  template<typename T>
  inline void expose()
  {
    if(not register_symbolic_link_to_registered_type<T>())
      internal::call_expose<T>::run();
  }
}

#endif // ifndef __eigenpy_expose_hpp__
