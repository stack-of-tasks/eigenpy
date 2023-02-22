//
// Copyright (c) 2014-2023 CNRS INRIA
//

#ifndef __eigenpy_eigen_from_python_hpp__
#define __eigenpy_eigen_from_python_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/scalar-conversion.hpp"

namespace eigenpy {

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct expected_pytype_for_arg {};

template <typename MatType>
struct expected_pytype_for_arg<MatType, Eigen::MatrixBase<MatType> > {
  static PyTypeObject const *get_pytype() {
    PyTypeObject const *py_type = eigenpy::getPyArrayType();
    return py_type;
  }
};

}  // namespace eigenpy

namespace boost {
namespace python {
namespace converter {

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct expected_pytype_for_arg<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> >
    : eigenpy::expected_pytype_for_arg<
          Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> > {};

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace eigenpy {
namespace details {
template <typename MatType, bool is_const = boost::is_const<MatType>::value>
struct copy_if_non_const {
  static void run(const Eigen::MatrixBase<MatType> &input,
                  PyArrayObject *pyArray) {
    EigenAllocator<MatType>::copy(input, pyArray);
  }
};

template <typename MatType>
struct copy_if_non_const<const MatType, true> {
  static void run(const Eigen::MatrixBase<MatType> & /*input*/,
                  PyArrayObject * /*pyArray*/) {}
};

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

template <typename _RefType>
struct referent_storage_eigen_ref {
  typedef _RefType RefType;
  typedef typename get_eigen_ref_plain_type<RefType>::type PlainObjectType;
  typedef typename ::eigenpy::aligned_storage<
      ::boost::python::detail::referent_size<RefType &>::value>::type
      AlignedStorage;

  referent_storage_eigen_ref()
      : pyArray(NULL),
        plain_ptr(NULL),
        ref_ptr(reinterpret_cast<RefType *>(ref_storage.bytes)) {}

  referent_storage_eigen_ref(const RefType &ref, PyArrayObject *pyArray,
                             PlainObjectType *plain_ptr = NULL)
      : pyArray(pyArray),
        plain_ptr(plain_ptr),
        ref_ptr(reinterpret_cast<RefType *>(ref_storage.bytes)) {
    Py_INCREF(pyArray);
    new (ref_storage.bytes) RefType(ref);
  }

  ~referent_storage_eigen_ref() {
    if (plain_ptr != NULL && PyArray_ISWRITEABLE(pyArray))
      copy_if_non_const<PlainObjectType>::run(*plain_ptr, pyArray);

    Py_DECREF(pyArray);

    if (plain_ptr != NULL) plain_ptr->~PlainObjectType();

    ref_ptr->~RefType();
  }

  AlignedStorage ref_storage;
  PyArrayObject *pyArray;
  PlainObjectType *plain_ptr;
  RefType *ref_ptr;
};
#endif

}  // namespace details
}  // namespace eigenpy

namespace boost {
namespace python {
namespace detail {
#if EIGEN_VERSION_AT_LEAST(3, 2, 0)
template <typename MatType, int Options, typename Stride>
struct referent_storage<Eigen::Ref<MatType, Options, Stride> &> {
  typedef Eigen::Ref<MatType, Options, Stride> RefType;
  typedef ::eigenpy::details::referent_storage_eigen_ref<RefType> StorageType;
  typedef typename ::eigenpy::aligned_storage<
      referent_size<StorageType &>::value>::type type;
};

template <typename MatType, int Options, typename Stride>
struct referent_storage<const Eigen::Ref<const MatType, Options, Stride> &> {
  typedef Eigen::Ref<const MatType, Options, Stride> RefType;
  typedef ::eigenpy::details::referent_storage_eigen_ref<RefType> StorageType;
  typedef typename ::eigenpy::aligned_storage<
      referent_size<StorageType &>::value>::type type;
};
#endif
}  // namespace detail
}  // namespace python
}  // namespace boost

namespace boost {
namespace python {
namespace converter {

#define EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(type)                       \
  typedef ::eigenpy::rvalue_from_python_data<type> Base;                 \
                                                                         \
  rvalue_from_python_data(rvalue_from_python_stage1_data const &_stage1) \
      : Base(_stage1) {}                                                 \
                                                                         \
  rvalue_from_python_data(void *convertible) : Base(convertible){};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct rvalue_from_python_data<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> const &>
    : ::eigenpy::rvalue_from_python_data<Eigen::Matrix<
          Scalar, Rows, Cols, Options, MaxRows, MaxCols> const &> {
  typedef Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> T;
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(T const &)
};

template <typename Derived>
struct rvalue_from_python_data<Eigen::MatrixBase<Derived> const &>
    : ::eigenpy::rvalue_from_python_data<Derived const &> {
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
};

template <typename Derived>
struct rvalue_from_python_data<Eigen::EigenBase<Derived> const &>
    : ::eigenpy::rvalue_from_python_data<Derived const &> {
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
};

template <typename Derived>
struct rvalue_from_python_data<Eigen::PlainObjectBase<Derived> const &>
    : ::eigenpy::rvalue_from_python_data<Derived const &> {
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
};

template <typename MatType, int Options, typename Stride>
struct rvalue_from_python_data<Eigen::Ref<MatType, Options, Stride> &>
    : rvalue_from_python_storage<Eigen::Ref<MatType, Options, Stride> &> {
  typedef Eigen::Ref<MatType, Options, Stride> RefType;

#if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) &&                        \
    (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) &&                 \
    (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) &&                   \
    !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing \
                                       this */
  // This must always be a POD struct with m_data its first member.
  BOOST_STATIC_ASSERT(BOOST_PYTHON_OFFSETOF(rvalue_from_python_storage<RefType>,
                                            stage1) == 0);
#endif

  // The usual constructor
  rvalue_from_python_data(rvalue_from_python_stage1_data const &_stage1) {
    this->stage1 = _stage1;
  }

  // This constructor just sets m_convertible -- used by
  // implicitly_convertible<> to perform the final step of the
  // conversion, where the construct() function is already known.
  rvalue_from_python_data(void *convertible) {
    this->stage1.convertible = convertible;
  }

  // Destroys any object constructed in the storage.
  ~rvalue_from_python_data() {
    typedef ::eigenpy::details::referent_storage_eigen_ref<RefType> StorageType;
    if (this->stage1.convertible == this->storage.bytes)
      static_cast<StorageType *>((void *)this->storage.bytes)->~StorageType();
  }
};

template <typename MatType, int Options, typename Stride>
struct rvalue_from_python_data<
    const Eigen::Ref<const MatType, Options, Stride> &>
    : rvalue_from_python_storage<
          const Eigen::Ref<const MatType, Options, Stride> &> {
  typedef Eigen::Ref<const MatType, Options, Stride> RefType;

#if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) &&                        \
    (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) &&                 \
    (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) &&                   \
    !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing \
                                       this */
  // This must always be a POD struct with m_data its first member.
  BOOST_STATIC_ASSERT(BOOST_PYTHON_OFFSETOF(rvalue_from_python_storage<RefType>,
                                            stage1) == 0);
#endif

  // The usual constructor
  rvalue_from_python_data(rvalue_from_python_stage1_data const &_stage1) {
    this->stage1 = _stage1;
  }

  // This constructor just sets m_convertible -- used by
  // implicitly_convertible<> to perform the final step of the
  // conversion, where the construct() function is already known.
  rvalue_from_python_data(void *convertible) {
    this->stage1.convertible = convertible;
  }

  // Destroys any object constructed in the storage.
  ~rvalue_from_python_data() {
    typedef ::eigenpy::details::referent_storage_eigen_ref<RefType> StorageType;
    if (this->stage1.convertible == this->storage.bytes)
      static_cast<StorageType *>((void *)this->storage.bytes)->~StorageType();
  }
};

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace eigenpy {

template <typename MatOrRefType>
void eigen_from_py_construct(
    PyObject *pyObj, bp::converter::rvalue_from_python_stage1_data *memory) {
  PyArrayObject *pyArray = reinterpret_cast<PyArrayObject *>(pyObj);
  assert((PyArray_DIMS(pyArray)[0] < INT_MAX) &&
         (PyArray_DIMS(pyArray)[1] < INT_MAX));

  bp::converter::rvalue_from_python_storage<MatOrRefType> *storage =
      reinterpret_cast<
          bp::converter::rvalue_from_python_storage<MatOrRefType> *>(
          reinterpret_cast<void *>(memory));

  EigenAllocator<MatOrRefType>::allocate(pyArray, storage);

  memory->convertible = storage->storage.bytes;
}

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct eigen_from_py_impl {
  typedef typename EigenType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj);

  /// \brief Allocate memory and copy pyObj in the new storage
  static void construct(PyObject *pyObj,
                        bp::converter::rvalue_from_python_stage1_data *memory);

  static void registration();
};

template <typename MatType>
struct eigen_from_py_impl<MatType, Eigen::MatrixBase<MatType> > {
  typedef typename MatType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj);

  /// \brief Allocate memory and copy pyObj in the new storage
  static void construct(PyObject *pyObj,
                        bp::converter::rvalue_from_python_stage1_data *memory);

  static void registration();
};

#ifdef EIGENPY_MSVC_COMPILER
template <typename EigenType>
struct EigenFromPy<EigenType,
                   typename boost::remove_reference<EigenType>::type::Scalar>
#else
template <typename EigenType, typename _Scalar>
struct EigenFromPy
#endif
    : eigen_from_py_impl<EigenType> {
};

template <typename MatType>
void *eigen_from_py_impl<MatType, Eigen::MatrixBase<MatType> >::convertible(
    PyObject *pyObj) {
  if (!call_PyArray_Check(reinterpret_cast<PyObject *>(pyObj))) return 0;

  PyArrayObject *pyArray = reinterpret_cast<PyArrayObject *>(pyObj);

  if (!np_type_is_convertible_into_scalar<Scalar>(
          EIGENPY_GET_PY_ARRAY_TYPE(pyArray)))
    return 0;

  if (MatType::IsVectorAtCompileTime) {
    const Eigen::DenseIndex size_at_compile_time =
        MatType::IsRowMajor ? MatType::ColsAtCompileTime
                            : MatType::RowsAtCompileTime;

    switch (PyArray_NDIM(pyArray)) {
      case 0:
        return 0;
      case 1: {
        if (size_at_compile_time != Eigen::Dynamic) {
          // check that the sizes at compile time matche
          if (PyArray_DIMS(pyArray)[0] == size_at_compile_time)
            return pyArray;
          else
            return 0;
        } else  // This is a dynamic MatType
          return pyArray;
      }
      case 2: {
        // Special care of scalar matrix of dimension 1x1.
        if (PyArray_DIMS(pyArray)[0] == 1 && PyArray_DIMS(pyArray)[1] == 1) {
          if (size_at_compile_time != Eigen::Dynamic) {
            if (size_at_compile_time == 1)
              return pyArray;
            else
              return 0;
          } else  // This is a dynamic MatType
            return pyArray;
        }

        if (PyArray_DIMS(pyArray)[0] > 1 && PyArray_DIMS(pyArray)[1] > 1) {
          return 0;
        }

        if (((PyArray_DIMS(pyArray)[0] == 1) &&
             (MatType::ColsAtCompileTime == 1)) ||
            ((PyArray_DIMS(pyArray)[1] == 1) &&
             (MatType::RowsAtCompileTime == 1))) {
          return 0;
        }

        if (size_at_compile_time !=
            Eigen::Dynamic) {  // This is a fixe size vector
          const Eigen::DenseIndex pyArray_size =
              PyArray_DIMS(pyArray)[0] > PyArray_DIMS(pyArray)[1]
                  ? PyArray_DIMS(pyArray)[0]
                  : PyArray_DIMS(pyArray)[1];
          if (size_at_compile_time != pyArray_size) return 0;
        }
        break;
      }
      default:
        return 0;
    }
  } else  // this is a matrix
  {
    if (PyArray_NDIM(pyArray) ==
        1)  // We can always convert a vector into a matrix
    {
      return pyArray;
    }

    if (PyArray_NDIM(pyArray) != 2) {
      return 0;
    }

    if (PyArray_NDIM(pyArray) == 2) {
      const int R = (int)PyArray_DIMS(pyArray)[0];
      const int C = (int)PyArray_DIMS(pyArray)[1];

      if ((MatType::RowsAtCompileTime != R) &&
          (MatType::RowsAtCompileTime != Eigen::Dynamic))
        return 0;
      if ((MatType::ColsAtCompileTime != C) &&
          (MatType::ColsAtCompileTime != Eigen::Dynamic))
        return 0;
    }
  }

#ifdef NPY_1_8_API_VERSION
  if (!(PyArray_FLAGS(pyArray)))
#else
  if (!(PyArray_FLAGS(pyArray) & NPY_ALIGNED))
#endif
  {
    return 0;
  }

  return pyArray;
}

template <typename MatType>
void eigen_from_py_impl<MatType, Eigen::MatrixBase<MatType> >::construct(
    PyObject *pyObj, bp::converter::rvalue_from_python_stage1_data *memory) {
  eigen_from_py_construct<MatType>(pyObj, memory);
}

template <typename MatType>
void eigen_from_py_impl<MatType, Eigen::MatrixBase<MatType> >::registration() {
  bp::converter::registry::push_back(
      reinterpret_cast<void *(*)(_object *)>(&eigen_from_py_impl::convertible),
      &eigen_from_py_impl::construct, bp::type_id<MatType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                          ,
      &eigenpy::expected_pytype_for_arg<MatType>::get_pytype
#endif
  );
}

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct eigen_from_py_converter_impl;

template <typename EigenType>
struct EigenFromPyConverter : eigen_from_py_converter_impl<EigenType> {};

template <typename MatType>
struct eigen_from_py_converter_impl<MatType, Eigen::MatrixBase<MatType> > {
  static void registration() {
    EigenFromPy<MatType>::registration();

    // Add conversion to Eigen::MatrixBase<MatType>
    typedef Eigen::MatrixBase<MatType> MatrixBase;
    EigenFromPy<MatrixBase>::registration();

    // Add conversion to Eigen::EigenBase<MatType>
    typedef Eigen::EigenBase<MatType> EigenBase;
    EigenFromPy<EigenBase, typename MatType::Scalar>::registration();

    // Add conversion to Eigen::PlainObjectBase<MatType>
    typedef Eigen::PlainObjectBase<MatType> PlainObjectBase;
    EigenFromPy<PlainObjectBase>::registration();

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)
    // Add conversion to Eigen::Ref<MatType>
    typedef Eigen::Ref<MatType> RefType;
    EigenFromPy<RefType>::registration();

    // Add conversion to Eigen::Ref<MatType>
    typedef const Eigen::Ref<const MatType> ConstRefType;
    EigenFromPy<ConstRefType>::registration();
#endif
  }
};

template <typename MatType>
struct EigenFromPy<Eigen::MatrixBase<MatType> > : EigenFromPy<MatType> {
  typedef EigenFromPy<MatType> EigenFromPyDerived;
  typedef Eigen::MatrixBase<MatType> Base;

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &EigenFromPy::construct, bp::type_id<Base>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                     ,
        &eigenpy::expected_pytype_for_arg<MatType>::get_pytype
#endif
    );
  }
};

template <typename MatType>
struct EigenFromPy<Eigen::EigenBase<MatType>, typename MatType::Scalar>
    : EigenFromPy<MatType> {
  typedef EigenFromPy<MatType> EigenFromPyDerived;
  typedef Eigen::EigenBase<MatType> Base;

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &EigenFromPy::construct, bp::type_id<Base>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                     ,
        &eigenpy::expected_pytype_for_arg<MatType>::get_pytype
#endif
    );
  }
};

template <typename MatType>
struct EigenFromPy<Eigen::PlainObjectBase<MatType> > : EigenFromPy<MatType> {
  typedef EigenFromPy<MatType> EigenFromPyDerived;
  typedef Eigen::PlainObjectBase<MatType> Base;

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &EigenFromPy::construct, bp::type_id<Base>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                     ,
        &eigenpy::expected_pytype_for_arg<MatType>::get_pytype
#endif
    );
  }
};

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

template <typename MatType, int Options, typename Stride>
struct EigenFromPy<Eigen::Ref<MatType, Options, Stride> > {
  typedef Eigen::Ref<MatType, Options, Stride> RefType;
  typedef typename MatType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj) {
    if (!call_PyArray_Check(pyObj)) return 0;
    PyArrayObject *pyArray = reinterpret_cast<PyArrayObject *>(pyObj);
    if (!PyArray_ISWRITEABLE(pyArray)) return 0;
    return EigenFromPy<MatType>::convertible(pyObj);
  }

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &eigen_from_py_construct<RefType>, bp::type_id<RefType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                               ,
        &eigenpy::expected_pytype_for_arg<MatType>::get_pytype
#endif
    );
  }
};

template <typename MatType, int Options, typename Stride>
struct EigenFromPy<const Eigen::Ref<const MatType, Options, Stride> > {
  typedef const Eigen::Ref<const MatType, Options, Stride> ConstRefType;
  typedef typename MatType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj) {
    return EigenFromPy<MatType>::convertible(pyObj);
  }

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &eigen_from_py_construct<ConstRefType>, bp::type_id<ConstRefType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                                    ,
        &eigenpy::expected_pytype_for_arg<MatType>::get_pytype
#endif
    );
  }
};
#endif

}  // namespace eigenpy

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
#include "eigenpy/tensor/eigen-from-python.hpp"
#endif

#endif  // __eigenpy_eigen_from_python_hpp__
