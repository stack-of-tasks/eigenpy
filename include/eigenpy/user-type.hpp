//
// Copyright (c) 2020-2022 INRIA
//

#ifndef __eigenpy_user_type_hpp__
#define __eigenpy_user_type_hpp__

#include <iostream>

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/register.hpp"
#include "eigenpy/user-type-traits.hpp"

namespace eigenpy {
/// \brief Default cast algo to cast a From to To. Can be specialized for any
/// types.
template <typename From, typename To>
struct cast {
  static To run(const From& from) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
    return static_cast<To>(from);
#pragma GCC diagnostic pop
  }
};

namespace internal {

template <typename From, typename To>
static void cast(void* from_, void* to_, npy_intp n, void* /*fromarr*/,
                 void* /*toarr*/) {
  //      std::cout << "cast::run" << std::endl;
  const From* from = static_cast<From*>(from_);
  To* to = static_cast<To*>(to_);
  if (user_type_traits<From>::requires_initialization) {
    const From zero(0);
    for (npy_intp i = 0; i < n; i++) {
      to[i] = eigenpy::cast<From, To>::run(value_or_zero(from[i], zero));
    }
  } else {
    for (npy_intp i = 0; i < n; i++) {
      to[i] = eigenpy::cast<From, To>::run(from[i]);
    }
  }
}

template <typename T>
struct getitem {
  ///
  /// \brief Get a python object from an array
  ///        It returns a standard Python object from
  ///        a single element of the array object arr pointed to by data.
  /// \param[in] data Pointer to the first element of the C++ data stream
  /// \param[in] arr  Pointer to the first element of the Python object data
  /// stream
  ///
  /// \returns PyObject corresponding to the python datastream.
  ///
  static PyObject* run(void* data, void* /* arr */) {
    //    std::cout << "getitem" << std::endl;
    T* elt_ptr = static_cast<T*>(data);
    if (user_type_traits<T>::requires_initialization &&
        user_type_traits<T>::is_uninitialized(*elt_ptr)) {
      // Heal the never-constructed (zero-filled) slot before handing it to
      // Python, which may keep a reference to it.
      *elt_ptr = T(0);
    }
    bp::object m(boost::ref(*elt_ptr));
    Py_INCREF(m.ptr());
    return m.ptr();
  }
};

template <typename T, int type_code = NumpyEquivalentType<T>::type_code>
struct SpecialMethods {
  inline static void copyswap(void* /*dst*/, void* /*src*/, int /*swap*/,
                              void* /*arr*/) /*{}*/;
  inline static PyObject* getitem(void* /*ip*/,
                                  void* /*ap*/) /*{ return NULL; }*/;
  inline static int setitem(PyObject* /*op*/, void* /*ov*/,
                            void* /*ap*/) /*{ return -1; }*/;
  inline static void copyswapn(void* /*dest*/, long /*dstride*/, void* /*src*/,
                               long /*sstride*/, long /*n*/, int /*swap*/,
                               void* /*arr*/) /*{}*/;
  inline static npy_bool nonzero(
      void* /*ip*/, void* /*array*/) /*{ return (npy_bool)false; }*/;
  inline static void dotfunc(void* /*ip0_*/, npy_intp /*is0*/, void* /*ip1_*/,
                             npy_intp /*is1*/, void* /*op*/, npy_intp /*n*/,
                             void* /*arr*/);
  inline static int fill(void* data_, npy_intp length, void* arr);
  inline static int fillwithscalar(void* buffer_, npy_intp length, void* value,
                                   void* arr);
};

template <typename T>
struct OffsetOf {
  struct Data {
    char c;
    T v;
  };

  enum { value = offsetof(Data, v) };
};

template <typename T>
struct SpecialMethods<T, NPY_USERDEF> {
  static void copyswap(void* dst, void* src, int swap, void* /*arr*/) {
    //    std::cout << "copyswap" << std::endl;
    if (user_type_traits<T>::requires_initialization) {
      // Slots may be zero-filled but never constructed: heal them before
      // they are read (assignment into such a slot is fine by the
      // user_type_traits contract).
      if (src != NULL) {
        T& t2 = *static_cast<T*>(src);
        if (user_type_traits<T>::is_uninitialized(t2)) t2 = T(0);
        T& t1 = *static_cast<T*>(dst);
        t1 = t2;
      }

      if (swap) {
        T& t1 = *static_cast<T*>(dst);
        T& t2 = *static_cast<T*>(src);
        if (user_type_traits<T>::is_uninitialized(t1)) t1 = T(0);
        if (user_type_traits<T>::is_uninitialized(t2)) t2 = T(0);
        std::swap(t1, t2);
      }
    } else {
      if (src != NULL) {
        T& t1 = *static_cast<T*>(dst);
        T& t2 = *static_cast<T*>(src);
        t1 = t2;
      }

      if (swap) {
        T& t1 = *static_cast<T*>(dst);
        T& t2 = *static_cast<T*>(src);
        std::swap(t1, t2);
      }
    }
  }

  static PyObject* getitem(void* ip, void* ap) {
    return eigenpy::internal::getitem<T>::run(ip, ap);
  }

  ///
  /// \brief Set a python object in an array.
  ///        It sets the Python object "item" into the array, arr, at the
  ///        position pointed to by data. This function deals with “misbehaved”
  ///        arrays. If successful, a zero is returned, otherwise, a negative
  ///        one is returned (and a Python error set).

  /// \param[in] src_obj  Pointer to the location of the python object
  /// \param[in] dest_ptr Pointer to the location in the array where the source
  /// object should be saved. \param[in] array Pointer to the location of the
  /// array
  ///
  /// \returns int Success(0) or Failure(-1)
  ///

  inline static int setitem(PyObject* src_obj, void* dest_ptr, void* array) {
    //    std::cout << "setitem" << std::endl;
    if (array == NULL) {
      eigenpy::Exception("Cannot retrieve the type stored in the array.");
      return -1;
    }

    PyArrayObject* py_array = static_cast<PyArrayObject*>(array);
    PyArray_Descr* descr = PyArray_DTYPE(py_array);
    PyTypeObject* array_scalar_type = descr->typeobj;
    PyTypeObject* src_obj_type = Py_TYPE(src_obj);

    // For types whose all-zero pattern is the "never constructed" state,
    // zero the destination first so that T::operator= takes its
    // init-before-set path even if the slot holds garbage. Note that numpy
    // never destructs user-dtype elements, so overwriting a previously
    // constructed element this way trades a crash for a bounded leak.
    prepare_destination<T>(dest_ptr);

    T& dest = *static_cast<T*>(dest_ptr);
    if (array_scalar_type != src_obj_type) {
      long long src_value = PyLong_AsLongLong(src_obj);
      if (src_value == -1 && PyErr_Occurred()) {
        std::stringstream ss;
        ss << "The input type is of wrong type. ";
        ss << "The expected type is " << bp::type_info(typeid(T)).name()
           << std::endl;
        eigenpy::Exception(ss.str());
        return -1;
      }

      dest = T(src_value);

    } else {
      bp::extract<T&> extract_src_obj(src_obj);
      if (!extract_src_obj.check()) {
        std::cout << "if (!extract_src_obj.check())" << std::endl;
        std::stringstream ss;
        ss << "The input type is of wrong type. ";
        ss << "The expected type is " << bp::type_info(typeid(T)).name()
           << std::endl;
        eigenpy::Exception(ss.str());
        return -1;
      }

      const T& src = extract_src_obj();
      T& dest = *static_cast<T*>(dest_ptr);
      dest = src;
    }

    return 0;
  }

  inline static void copyswapn(void* dst, long dstride, void* src, long sstride,
                               long n, int swap, void* array) {
    //    std::cout << "copyswapn" << std::endl;

    char* dstptr = static_cast<char*>(dst);
    char* srcptr = static_cast<char*>(src);

    PyArrayObject* py_array = static_cast<PyArrayObject*>(array);
    PyArray_CopySwapFunc* copyswap =
        PyDataType_GetArrFuncs(PyArray_DESCR(py_array))->copyswap;

    for (npy_intp i = 0; i < n; i++) {
      copyswap(dstptr, srcptr, swap, array);
      dstptr += dstride;
      srcptr += sstride;
    }
  }

  inline static npy_bool nonzero(void* ip, void* array) {
    //    std::cout << "nonzero" << std::endl;
    static const T ZeroValue = T(0);
    PyArrayObject* py_array = static_cast<PyArrayObject*>(array);
    if (py_array == NULL || PyArray_ISBEHAVED_RO(py_array)) {
      const T& value = value_or_zero(*static_cast<T*>(ip), ZeroValue);
      return (npy_bool)(value != ZeroValue);
    } else {
      T tmp_value;
      PyDataType_GetArrFuncs(PyArray_DESCR(py_array))
          ->copyswap(&tmp_value, ip, PyArray_ISBYTESWAPPED(py_array), array);
      return (npy_bool)(tmp_value != ZeroValue);
    }
  }

  inline static void dotfunc(void* ip0_, npy_intp is0, void* ip1_, npy_intp is1,
                             void* op, npy_intp n, void* /*arr*/) {
    //    std::cout << "dotfunc" << std::endl;
    if (user_type_traits<T>::requires_initialization) {
      // Same conjugating semantics as Eigen's dot below, but reading each
      // operand through value_or_zero so that never-constructed (zero-filled)
      // slots are treated as an exact T(0) instead of being handed to the
      // scalar's operators.
      const T zero(0);
      T acc(0);
      const char* p0 = static_cast<const char*>(ip0_);
      const char* p1 = static_cast<const char*>(ip1_);
      for (npy_intp i = 0; i < n; ++i) {
        const T& x = *reinterpret_cast<const T*>(p0);
        const T& y = *reinterpret_cast<const T*>(p1);
        acc += Eigen::numext::conj(value_or_zero(x, zero)) *
               value_or_zero(y, zero);
        p0 += is0;
        p1 += is1;
      }
      *static_cast<T*>(op) = acc;
    } else {
      typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorT;
      typedef Eigen::InnerStride<Eigen::Dynamic> InputStride;
      typedef const Eigen::Map<const VectorT, 0, InputStride> ConstMapType;

      ConstMapType v0(static_cast<T*>(ip0_), n,
                      InputStride(is0 / (Eigen::DenseIndex)sizeof(T))),
          v1(static_cast<T*>(ip1_), n,
             InputStride(is1 / (Eigen::DenseIndex)sizeof(T)));

      *static_cast<T*>(op) = v0.dot(v1);
    }
  }

  inline static int fillwithscalar(void* buffer_, npy_intp length, void* value,
                                   void* /*arr*/) {
    //    std::cout << "fillwithscalar" << std::endl;
    if (user_type_traits<T>::requires_initialization) {
      const T zero(0);
      T r = value_or_zero(*static_cast<T*>(value), zero);
      T* buffer = static_cast<T*>(buffer_);
      npy_intp i;
      for (i = 0; i < length; i++) {
        buffer[i] = r;
      }
    } else {
      T r = *static_cast<T*>(value);
      T* buffer = static_cast<T*>(buffer_);
      npy_intp i;
      for (i = 0; i < length; i++) {
        buffer[i] = r;
      }
    }
    return 0;
  }

  static int fill(void* data_, npy_intp length, void* /*arr*/) {
    //    std::cout << "fill" << std::endl;
    T* data = static_cast<T*>(data_);
    if (user_type_traits<T>::requires_initialization) {
      const T zero(0);
      const T& first = value_or_zero(data[0], zero);
      const T& second = value_or_zero(data[1], zero);
      const T delta = second - first;
      T r = second;
      npy_intp i;
      for (i = 2; i < length; i++) {
        r = r + delta;
        data[i] = r;
      }
    } else {
      const T delta = data[1] - data[0];
      T r = data[1];
      npy_intp i;
      for (i = 2; i < length; i++) {
        r = r + delta;
        data[i] = r;
      }
    }
    return 0;
  }

};  //     struct SpecialMethods<T,NPY_USERDEF>

}  // namespace internal

template <typename From, typename To>
bool registerCast(const bool safe) {
  PyArray_Descr* from_array_descr = Register::getPyArrayDescr<From>();
  //    int from_typenum = Register::getTypeCode<From>();

  //    PyTypeObject * to_py_type = Register::getPyType<To>();
  int to_typenum = Register::getTypeCode<To>();
  assert(to_typenum >= 0 && "to_typenum is not valid");
  assert(from_array_descr != NULL && "from_array_descr is not valid");

  //    std::cout << "From: " << bp::type_info(typeid(From)).name() << " " <<
  //    Register::getTypeCode<From>()
  //    << " to: " << bp::type_info(typeid(To)).name() << " " <<
  //    Register::getTypeCode<To>()
  //    << "\n to_typenum: " << to_typenum
  //    << std::endl;

  if (call_PyArray_RegisterCastFunc(from_array_descr, to_typenum,
                                    static_cast<PyArray_VectorUnaryFunc*>(
                                        &eigenpy::internal::cast<From, To>)) <
      0) {
    std::stringstream ss;
    ss << "PyArray_RegisterCastFunc of the cast from "
       << bp::type_info(typeid(From)).name() << " to "
       << bp::type_info(typeid(To)).name() << " has failed.";
    eigenpy::Exception(ss.str());
    return false;
  }

  if (safe && call_PyArray_RegisterCanCast(from_array_descr, to_typenum,
                                           NPY_NOSCALAR) < 0) {
    std::stringstream ss;
    ss << "PyArray_RegisterCanCast of the cast from "
       << bp::type_info(typeid(From)).name() << " to "
       << bp::type_info(typeid(To)).name() << " has failed.";
    eigenpy::Exception(ss.str());
    return false;
  }

  return true;
}

/// \brief Get the class object for a wrapped type that has been exposed
///        through Boost.Python.
template <typename T>
boost::python::object getInstanceClass() {
  // Query into the registry for type T.
  bp::type_info type = bp::type_id<T>();
  const bp::converter::registration* registration =
      bp::converter::registry::query(type);

  // If the class is not registered, return None.
  if (!registration) {
    // std::cerr<<"Class Not Registered. Returning Empty."<<std::endl;
    return bp::object();
  }

  bp::handle<PyTypeObject> handle(
      bp::borrowed(registration->get_class_object()));
  return bp::object(handle);
}

template <typename Scalar>
int registerNewType(PyTypeObject* py_type_ptr = NULL) {
  // Check whether the type is a Numpy native type.
  // In this case, the registration is not required.
  if (isNumpyNativeType<Scalar>())
    return NumpyEquivalentType<Scalar>::type_code;

  // Retrieve the registered type for the current Scalar
  if (py_type_ptr == NULL) {  // retrive the type from Boost.Python
    py_type_ptr = Register::getPyType<Scalar>();
  }

  if (Register::isRegistered(py_type_ptr))
    return Register::getTypeCode(
        py_type_ptr);  // the type is already registered

  PyArray_GetItemFunc* getitem = &internal::SpecialMethods<Scalar>::getitem;
  PyArray_SetItemFunc* setitem = &internal::SpecialMethods<Scalar>::setitem;
  PyArray_NonzeroFunc* nonzero = &internal::SpecialMethods<Scalar>::nonzero;
  PyArray_CopySwapFunc* copyswap = &internal::SpecialMethods<Scalar>::copyswap;
  PyArray_CopySwapNFunc* copyswapn = reinterpret_cast<PyArray_CopySwapNFunc*>(
      &internal::SpecialMethods<Scalar>::copyswapn);
  PyArray_DotFunc* dotfunc = &internal::SpecialMethods<Scalar>::dotfunc;
  PyArray_FillFunc* fill = &internal::SpecialMethods<Scalar>::fill;
  PyArray_FillWithScalarFunc* fillwithscalar =
      &internal::SpecialMethods<Scalar>::fillwithscalar;

  int code = Register::registerNewType(
      py_type_ptr, &typeid(Scalar), sizeof(Scalar),
      internal::OffsetOf<Scalar>::value, getitem, setitem, nonzero, copyswap,
      copyswapn, dotfunc, fill, fillwithscalar);

  call_PyArray_RegisterCanCast(call_PyArray_DescrFromType(NPY_OBJECT), code,
                               NPY_NOSCALAR);

  return code;
}

}  // namespace eigenpy

#endif  // __eigenpy_user_type_hpp__
