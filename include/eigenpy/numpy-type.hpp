/*
 * Copyright 2018-2023 INRIA
 */

#ifndef __eigenpy_numpy_type_hpp__
#define __eigenpy_numpy_type_hpp__

#include <sstream>
#include <stdexcept>
#include <typeinfo>

#include "eigenpy/fwd.hpp"
#include "eigenpy/register.hpp"
#include "eigenpy/scalar-conversion.hpp"

namespace eigenpy {

template <typename Scalar>
bool np_type_is_convertible_into_scalar(const int np_type) {
  const auto scalar_np_code =
      static_cast<NPY_TYPES>(NumpyEquivalentType<Scalar>::type_code);

  if (scalar_np_code >= NPY_USERDEF)
    return np_type == Register::getTypeCode<Scalar>();

  if (scalar_np_code == np_type) return true;

  // Manage type promotion
  switch (np_type) {
    case NPY_BOOL:
      return FromTypeToType<bool, Scalar>::value;
    case NPY_INT8:
      return FromTypeToType<int8_t, Scalar>::value;
    case NPY_INT16:
      return FromTypeToType<int16_t, Scalar>::value;
    case NPY_INT32:
      return FromTypeToType<int32_t, Scalar>::value;
    case NPY_INT64:
      return FromTypeToType<int64_t, Scalar>::value;
    case NPY_UINT8:
      return FromTypeToType<uint8_t, Scalar>::value;
    case NPY_UINT16:
      return FromTypeToType<uint16_t, Scalar>::value;
    case NPY_UINT32:
      return FromTypeToType<uint32_t, Scalar>::value;
    case NPY_UINT64:
      return FromTypeToType<uint64_t, Scalar>::value;

#if defined _WIN32 || defined __CYGWIN__
    // Manage NPY_INT on Windows (NPY_INT32 is NPY_LONG).
    // See https://github.com/stack-of-tasks/eigenpy/pull/455
    case NPY_INT:
      return FromTypeToType<int32_t, Scalar>::value;
    case NPY_UINT:
      return FromTypeToType<uint32_t, Scalar>::value;
#endif  // WIN32

#if defined __APPLE__
    // Manage NPY_LONGLONG on Mac (NPY_INT64 is NPY_LONG)..
    // long long and long are both the same type
    // but NPY_LONGLONG and NPY_LONG are different dtype.
    // See https://github.com/stack-of-tasks/eigenpy/pull/455
    case NPY_LONGLONG:
      return FromTypeToType<int64_t, Scalar>::value;
    case NPY_ULONGLONG:
      return FromTypeToType<uint64_t, Scalar>::value;
#endif  // MAC
    case NPY_FLOAT:
      return FromTypeToType<float, Scalar>::value;
    case NPY_CFLOAT:
      return FromTypeToType<std::complex<float>, Scalar>::value;
    case NPY_DOUBLE:
      return FromTypeToType<double, Scalar>::value;
    case NPY_CDOUBLE:
      return FromTypeToType<std::complex<double>, Scalar>::value;
    case NPY_LONGDOUBLE:
      return FromTypeToType<long double, Scalar>::value;
    case NPY_CLONGDOUBLE:
      return FromTypeToType<std::complex<long double>, Scalar>::value;
    default:
      return false;
  }
}

struct EIGENPY_DLLAPI NumpyType {
  static NumpyType& getInstance();

  static bp::object make(PyArrayObject* pyArray, bool copy = false);

  static bp::object make(PyObject* pyObj, bool copy = false);

  static void sharedMemory(const bool value);

  static bool sharedMemory();

  static const PyTypeObject* getNumpyArrayType();

 protected:
  NumpyType();

  bp::object pyModule;

  // Numpy types
  bp::object NumpyArrayObject;
  PyTypeObject* NumpyArrayType;

  bool shared_memory;
};
}  // namespace eigenpy

#endif  // ifndef __eigenpy_numpy_type_hpp__
