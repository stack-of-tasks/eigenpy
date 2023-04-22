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
  if (static_cast<NPY_TYPES>(NumpyEquivalentType<Scalar>::type_code) >=
      NPY_USERDEF)
    return np_type == Register::getTypeCode<Scalar>();

  if (NumpyEquivalentType<Scalar>::type_code == np_type) return true;

  switch (np_type) {
    case NPY_INT:
      return FromTypeToType<int, Scalar>::value;
    case NPY_LONG:
      return FromTypeToType<long, Scalar>::value;
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

  static bp::object getNumpyType();

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
