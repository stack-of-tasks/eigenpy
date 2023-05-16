///
/// Copyright (c) 2023 CNRS INRIA
///
/// Definitions for exposing boost::optional<T> types.
/// Also works with std::optional.

#ifndef __eigenpy_optional_hpp__
#define __eigenpy_optional_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-from-python.hpp"
#include "eigenpy/registration.hpp"

#include <boost/optional.hpp>
#ifdef EIGENPY_WITH_CXX17_SUPPORT
#include <optional>
#endif

#ifndef EIGENPY_DEFAULT_OPTIONAL
#define EIGENPY_DEFAULT_OPTIONAL boost::optional
#endif

namespace boost {
namespace python {
namespace converter {

template <typename T>
struct expected_pytype_for_arg<boost::optional<T> >
    : expected_pytype_for_arg<T> {};

#ifdef EIGENPY_WITH_CXX17_SUPPORT
template <typename T>
struct expected_pytype_for_arg<std::optional<T> > : expected_pytype_for_arg<T> {
};
#endif

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace eigenpy {

namespace detail {

/// Helper struct to decide which type is the "none" type for a specific
/// optional<T> implementation.
template <template <typename> class OptionalTpl>
struct nullopt_helper {};

template <>
struct nullopt_helper<boost::optional> {
  typedef boost::none_t type;
  static type value() { return boost::none; }
};

#ifdef EIGENPY_WITH_CXX17_SUPPORT
template <>
struct nullopt_helper<std::optional> {
  typedef std::nullopt_t type;
  static type value() { return std::nullopt; }
};
#endif

template <typename NoneType>
struct NoneToPython {
  static PyObject *convert(const NoneType &) { Py_RETURN_NONE; }

  static void registration() {
    if (!check_registration<NoneType>()) {
      bp::to_python_converter<NoneType, NoneToPython, false>();
    }
  }
};

template <typename T,
          template <typename> class OptionalTpl = EIGENPY_DEFAULT_OPTIONAL>
struct OptionalToPython {
  static PyObject *convert(const OptionalTpl<T> &obj) {
    if (obj)
      return bp::incref(bp::object(*obj).ptr());
    else {
      return bp::incref(bp::object().ptr());  // None
    }
  }

  static PyTypeObject const *get_pytype() {
    return bp::converter::registered_pytype<T>::get_pytype();
  }

  static void registration() {
    if (!check_registration<OptionalTpl<T> >()) {
      bp::to_python_converter<OptionalTpl<T>, OptionalToPython, true>();
    }
  }
};

template <typename T,
          template <typename> class OptionalTpl = EIGENPY_DEFAULT_OPTIONAL>
struct OptionalFromPython {
  static void *convertible(PyObject *obj_ptr);

  static void construct(PyObject *obj_ptr,
                        bp::converter::rvalue_from_python_stage1_data *memory);

  static void registration();
};

template <typename T, template <typename> class OptionalTpl>
void *OptionalFromPython<T, OptionalTpl>::convertible(PyObject *obj_ptr) {
  if (obj_ptr == Py_None) {
    return obj_ptr;
  }
  bp::extract<T> bp_obj(obj_ptr);
  if (!bp_obj.check())
    return 0;
  else
    return obj_ptr;
}

template <typename T, template <typename> class OptionalTpl>
void OptionalFromPython<T, OptionalTpl>::construct(
    PyObject *obj_ptr, bp::converter::rvalue_from_python_stage1_data *memory) {
  // create storage
  using rvalue_storage_t =
      bp::converter::rvalue_from_python_storage<OptionalTpl<T> >;
  void *storage =
      reinterpret_cast<rvalue_storage_t *>(reinterpret_cast<void *>(memory))
          ->storage.bytes;

  if (obj_ptr == Py_None) {
    new (storage) OptionalTpl<T>(nullopt_helper<OptionalTpl>::value());
  } else {
    const T value = bp::extract<T>(obj_ptr);
    new (storage) OptionalTpl<T>(value);
  }

  memory->convertible = storage;
}

template <typename T, template <typename> class OptionalTpl>
void OptionalFromPython<T, OptionalTpl>::registration() {
  bp::converter::registry::push_back(
      &convertible, &construct, bp::type_id<OptionalTpl<T> >(),
      bp::converter::expected_pytype_for_arg<OptionalTpl<T> >::get_pytype);
}

}  // namespace detail

/// Register converters for the type `optional<T>` to Python.
/// By default \tparam optional is `EIGENPY_DEFAULT_OPTIONAL`.
template <typename T,
          template <typename> class OptionalTpl = EIGENPY_DEFAULT_OPTIONAL>
struct OptionalConverter {
  static void registration() {
    detail::OptionalToPython<T, OptionalTpl>::registration();
    detail::OptionalFromPython<T, OptionalTpl>::registration();
  }
};

}  // namespace eigenpy

#endif  // __eigenpy_optional_hpp__
