//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_utils_std_unique_ptr_hpp__
#define __eigenpy_utils_std_unique_ptr_hpp__

#include "eigenpy/fwd.hpp"

#include <boost/python.hpp>

#include <memory>
#include <iostream>

namespace eigenpy {

/// result_converter of StdUniquePtrCallPolicies
struct StdUniquePtrResultConverter {
  template <class T>
  struct apply {
    struct type {
      typedef typename bp::detail::value_arg<T>::type argument_type;

      /// TODO, this work by copy
      /// We maybe can transfer the ownership to Python for class type
      /// and when argument_type is an lvalue ref
      PyObject* operator()(argument_type x) const {
        return bp::to_python_value<const typename T::element_type&>()(*x);
      }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
      PyTypeObject const* get_pytype() const {
        return bp::to_python_value<const typename T::element_type&>()
            .get_pytype();
      }
#endif
      BOOST_STATIC_CONSTANT(bool, uses_registry = true);
    };
  };
};

/// Access CallPolicie to get std::unique_ptr value from a functio
/// that return an std::unique_ptr
struct StdUniquePtrCallPolicies : bp::default_call_policies {
  typedef StdUniquePtrResultConverter result_converter;
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_std_unique_ptr_hpp__
