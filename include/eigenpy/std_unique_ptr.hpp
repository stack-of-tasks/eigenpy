//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_utils_std_unique_ptr_hpp__
#define __eigenpy_utils_std_unique_ptr_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/utils/traits.hpp"

#include <boost/python.hpp>

#include <memory>

namespace eigenpy {

namespace details {

template <typename T>
typename std::enable_if<is_class_or_union_remove_cvref<T>::value,
                        PyObject*>::type
unique_ptr_to_python(std::unique_ptr<T>&& x) {
  if (!x) {
    return bp::detail::none();
  } else {
    return bp::detail::make_owning_holder::execute(x.release());
  }
}

template <typename T>
typename std::enable_if<!is_class_or_union_remove_cvref<T>::value,
                        PyObject*>::type
unique_ptr_to_python(std::unique_ptr<T>&& x) {
  if (!x) {
    return bp::detail::none();
  } else {
    return bp::to_python_value<const T&>()(*x);
  }
}

}  // namespace details

/// result_converter of StdUniquePtrCallPolicies
struct StdUniquePtrResultConverter {
  template <typename T>
  struct apply {
    struct type {
      typedef typename T::element_type element_type;

      PyObject* operator()(T&& x) const {
        return details::unique_ptr_to_python(std::forward<T>(x));
      }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
      PyTypeObject const* get_pytype() const {
        return bp::to_python_value<const element_type&>().get_pytype();
      }
#endif
      BOOST_STATIC_CONSTANT(bool, uses_registry = true);
    };
  };
};

/// CallPolicies to get std::unique_ptr value from a function
/// that return an std::unique_ptr.
/// If the object inside the std::unique_ptr is a class or an union
/// it will be moved. In other case, it will be copied.
struct StdUniquePtrCallPolicies : bp::default_call_policies {
  typedef StdUniquePtrResultConverter result_converter;
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_std_unique_ptr_hpp__
