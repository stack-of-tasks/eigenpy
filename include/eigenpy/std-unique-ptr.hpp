//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_utils_std_unique_ptr_hpp__
#define __eigenpy_utils_std_unique_ptr_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/utils/traits.hpp"
#include "eigenpy/utils/python-compat.hpp"

#include <boost/python.hpp>

#include <memory>
#include <type_traits>

namespace eigenpy {

namespace details {

/// Transfer std::unique_ptr ownership to an owning holder
template <typename T>
typename std::enable_if<!is_python_primitive_type<T>::value, PyObject*>::type
unique_ptr_to_python(std::unique_ptr<T>&& x) {
  typedef bp::objects::pointer_holder<std::unique_ptr<T>, T> holder_t;
  if (!x) {
    return bp::detail::none();
  } else {
    return bp::objects::make_ptr_instance<T, holder_t>::execute(x);
  }
}

/// Convert and copy the primitive value to python
template <typename T>
typename std::enable_if<is_python_primitive_type<T>::value, PyObject*>::type
unique_ptr_to_python(std::unique_ptr<T>&& x) {
  if (!x) {
    return bp::detail::none();
  } else {
    return bp::to_python_value<const T&>()(*x);
  }
}

/// std::unique_ptr keep the ownership but a reference to the std::unique_ptr
/// value is created
template <typename T>
typename std::enable_if<!is_python_primitive_type<T>::value, PyObject*>::type
internal_unique_ptr_to_python(std::unique_ptr<T>& x) {
  if (!x) {
    return bp::detail::none();
  } else {
    return bp::detail::make_reference_holder::execute(x.get());
  }
}

/// Convert and copy the primitive value to python
template <typename T>
typename std::enable_if<is_python_primitive_type<T>::value, PyObject*>::type
internal_unique_ptr_to_python(std::unique_ptr<T>& x) {
  if (!x) {
    return bp::detail::none();
  } else {
    return bp::to_python_value<const T&>()(*x);
  }
}

/// result_converter of StdUniquePtrCallPolicies
struct StdUniquePtrResultConverter {
  template <typename T>
  struct apply {
    struct type {
      typedef typename T::element_type element_type;

      PyObject* operator()(T&& x) const {
        return unique_ptr_to_python(std::forward<T>(x));
      }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
      PyTypeObject const* get_pytype() const {
        return bp::to_python_value<const element_type&>().get_pytype();
      }
#endif
    };
  };
};

/// result_converter of ReturnInternalStdUniquePtr
struct InternalStdUniquePtrConverter {
  template <typename T>
  struct apply {
    struct type {
      typedef typename remove_cvref<T>::type::element_type element_type;

      PyObject* operator()(T x) const {
        return internal_unique_ptr_to_python(x);
      }
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
      PyTypeObject const* get_pytype() const {
        return bp::to_python_value<const element_type&>().get_pytype();
      }
#endif
    };
  };
};

}  // namespace details

/// CallPolicies to get std::unique_ptr value from a function
/// that return an std::unique_ptr.
/// If the object inside the std::unique_ptr is a class or an union
/// it will be moved. In other case, it will be copied.
struct StdUniquePtrCallPolicies : bp::default_call_policies {
  typedef details::StdUniquePtrResultConverter result_converter;
};

/// Variant of \see bp::return_internal_reference that extract std::unique_ptr
/// content reference before converting it into a PyObject
struct ReturnInternalStdUniquePtr : bp::return_internal_reference<> {
  typedef details::InternalStdUniquePtrConverter result_converter;

  template <class ArgumentPackage>
  static PyObject* postcall(ArgumentPackage const& args_, PyObject* result) {
    // Don't run return_internal_reference postcall on primitive type
    if (PyInt_Check(result) || PyBool_Check(result) || PyFloat_Check(result) ||
        PyStr_Check(result) || PyComplex_Check(result)) {
      return result;
    }
    return bp::return_internal_reference<>::postcall(args_, result);
  }
};

}  // namespace eigenpy

namespace boost {
namespace python {

/// Specialize to_python_value for std::unique_ptr
template <typename T>
struct to_python_value<const std::unique_ptr<T>&>
    : eigenpy::details::StdUniquePtrResultConverter::apply<
          std::unique_ptr<T>>::type {};

}  // namespace python
}  // namespace boost

#endif  // ifndef __eigenpy_utils_std_unique_ptr_hpp__
