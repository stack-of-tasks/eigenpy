/// Copyright (c) 2023 CNRS INRIA
/// Definitions for exposing boost::optional<T> types.
/// Also works with std::optional.

#ifndef __eigenpy_optional_hpp__
#define __eigenpy_optional_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-from-python.hpp"
#include <boost/optional.hpp>

#define EIGENPY_DEFAULT_OPTIONAL boost::optional

namespace boost {
namespace python {
namespace converter {

template <typename T>
struct expected_pytype_for_arg<EIGENPY_DEFAULT_OPTIONAL<T> >
    : expected_pytype_for_arg<T> {};

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace eigenpy {
namespace detail {

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
    bp::to_python_converter<OptionalTpl<T>, OptionalToPython, true>();
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
    new (storage) OptionalTpl<T>(boost::none);
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
