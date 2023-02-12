//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_register_hpp__
#define __eigenpy_register_hpp__

#include <algorithm>
#include <map>
#include <string>
#include <typeinfo>

#include "eigenpy/exception.hpp"
#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy.hpp"

namespace eigenpy {

/// \brief Structure collecting all the types registers in Numpy via EigenPy
struct EIGENPY_DLLAPI Register {
  static PyArray_Descr *getPyArrayDescr(PyTypeObject *py_type_ptr);

  template <typename Scalar>
  static bool isRegistered() {
    return isRegistered(Register::getPyType<Scalar>());
  }

  static bool isRegistered(PyTypeObject *py_type_ptr);

  static int getTypeCode(PyTypeObject *py_type_ptr);

  template <typename Scalar>
  static PyTypeObject *getPyType() {
    if (!isNumpyNativeType<Scalar>()) {
      const PyTypeObject *const_py_type_ptr =
          bp::converter::registered_pytype<Scalar>::get_pytype();
      if (const_py_type_ptr == NULL) {
        std::stringstream ss;
        ss << "The type " << typeid(Scalar).name()
           << " does not have a registered converter inside Boot.Python."
           << std::endl;
        throw std::invalid_argument(ss.str());
      }
      PyTypeObject *py_type_ptr = const_cast<PyTypeObject *>(const_py_type_ptr);
      return py_type_ptr;
    } else {
      PyArray_Descr *new_descr =
          call_PyArray_DescrFromType(NumpyEquivalentType<Scalar>::type_code);
      return new_descr->typeobj;
    }
  }

  template <typename Scalar>
  static PyArray_Descr *getPyArrayDescr() {
    if (!isNumpyNativeType<Scalar>()) {
      return getPyArrayDescr(getPyType<Scalar>());
    } else {
      return call_PyArray_DescrFromType(NumpyEquivalentType<Scalar>::type_code);
    }
  }

  template <typename Scalar>
  static int getTypeCode() {
    if (isNumpyNativeType<Scalar>())
      return NumpyEquivalentType<Scalar>::type_code;
    else {
      const std::type_info &info = typeid(Scalar);
      if (instance().type_to_py_type_bindings.find(&info) !=
          instance().type_to_py_type_bindings.end()) {
        PyTypeObject *py_type = instance().type_to_py_type_bindings[&info];
        int code = instance().py_array_code_bindings[py_type];

        return code;
      } else
        return -1;  // type not registered
    }
  }

  static int registerNewType(
      PyTypeObject *py_type_ptr, const std::type_info *type_info_ptr,
      const int type_size, const int alignment, PyArray_GetItemFunc *getitem,
      PyArray_SetItemFunc *setitem, PyArray_NonzeroFunc *nonzero,
      PyArray_CopySwapFunc *copyswap, PyArray_CopySwapNFunc *copyswapn,
      PyArray_DotFunc *dotfunc, PyArray_FillFunc *fill,
      PyArray_FillWithScalarFunc *fillwithscalar);

  static Register &instance();

 private:
  Register(){};

  struct Compare_PyTypeObject {
    bool operator()(const PyTypeObject *a, const PyTypeObject *b) const {
      return std::string(a->tp_name) < std::string(b->tp_name);
    }
  };

  struct Compare_TypeInfo {
    bool operator()(const std::type_info *a, const std::type_info *b) const {
      return std::string(a->name()) < std::string(b->name());
    }
  };

  typedef std::map<const std::type_info *, PyTypeObject *, Compare_TypeInfo>
      MapInfo;
  MapInfo type_to_py_type_bindings;

  typedef std::map<PyTypeObject *, PyArray_Descr *, Compare_PyTypeObject>
      MapDescr;
  MapDescr py_array_descr_bindings;

  typedef std::map<PyTypeObject *, int, Compare_PyTypeObject> MapCode;
  MapCode py_array_code_bindings;
};

}  // namespace eigenpy

#endif  // __eigenpy_register_hpp__
