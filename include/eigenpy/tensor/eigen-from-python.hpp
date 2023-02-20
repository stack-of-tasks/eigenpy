//
// Copyright (c) 2023 INRIA
//

#ifndef __eigenpy_tensor_eigen_from_python_hpp__
#define __eigenpy_tensor_eigen_from_python_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/scalar-conversion.hpp"

namespace eigenpy {

template <typename TensorType>
struct expected_pytype_for_arg<TensorType, Eigen::TensorBase<TensorType> > {
  static PyTypeObject const *get_pytype() {
    PyTypeObject const *py_type = eigenpy::getPyArrayType();
    return py_type;
  }
};

}  // namespace eigenpy

namespace boost {
namespace python {
namespace converter {

template <typename Scalar, int Rank, int Options, typename IndexType>
struct expected_pytype_for_arg<Eigen::Tensor<Scalar, Rank, Options, IndexType> >
    : eigenpy::expected_pytype_for_arg<
          Eigen::Tensor<Scalar, Rank, Options, IndexType> > {};

template <typename Scalar, int Rank, int Options, typename IndexType>
struct rvalue_from_python_data<
    Eigen::Tensor<Scalar, Rank, Options, IndexType> const &>
    : ::eigenpy::rvalue_from_python_data<
          Eigen::Tensor<Scalar, Rank, Options, IndexType> const &> {
  typedef Eigen::Tensor<Scalar, Rank, Options, IndexType> T;
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(T const &)
};

template <typename Derived>
struct rvalue_from_python_data<Eigen::TensorBase<Derived> const &>
    : ::eigenpy::rvalue_from_python_data<Derived const &> {
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
};

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace boost {
namespace python {
namespace detail {
template <typename TensorType>
struct referent_storage<Eigen::TensorRef<TensorType> &> {
  typedef Eigen::TensorRef<TensorType> RefType;
  typedef ::eigenpy::details::referent_storage_eigen_ref<RefType> StorageType;
  typedef typename ::eigenpy::aligned_storage<
      referent_size<StorageType &>::value>::type type;
};

template <typename TensorType>
struct referent_storage<const Eigen::TensorRef<const TensorType> &> {
  typedef Eigen::TensorRef<const TensorType> RefType;
  typedef ::eigenpy::details::referent_storage_eigen_ref<RefType> StorageType;
  typedef typename ::eigenpy::aligned_storage<
      referent_size<StorageType &>::value>::type type;
};
}  // namespace detail
}  // namespace python
}  // namespace boost

namespace eigenpy {

template <typename TensorType>
struct eigen_from_py_impl<TensorType, Eigen::TensorBase<TensorType> > {
  typedef typename TensorType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj);

  /// \brief Allocate memory and copy pyObj in the new storage
  static void construct(PyObject *pyObj,
                        bp::converter::rvalue_from_python_stage1_data *memory);

  static void registration();
};

template <typename TensorType>
void *
eigen_from_py_impl<TensorType, Eigen::TensorBase<TensorType> >::convertible(
    PyObject *pyObj) {
  if (!call_PyArray_Check(reinterpret_cast<PyObject *>(pyObj))) return 0;

  typedef typename Eigen::internal::traits<TensorType>::Index Index;
  static const Index NumIndices = TensorType::NumIndices;

  PyArrayObject *pyArray = reinterpret_cast<PyArrayObject *>(pyObj);

  if (!np_type_is_convertible_into_scalar<Scalar>(
          EIGENPY_GET_PY_ARRAY_TYPE(pyArray)))
    return 0;

  if (!(PyArray_NDIM(pyArray) == NumIndices || NumIndices == Eigen::Dynamic))
    return 0;

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

template <typename TensorType>
void eigen_from_py_impl<TensorType, Eigen::TensorBase<TensorType> >::construct(
    PyObject *pyObj, bp::converter::rvalue_from_python_stage1_data *memory) {
  eigen_from_py_construct<TensorType>(pyObj, memory);
}

template <typename TensorType>
void eigen_from_py_impl<TensorType,
                        Eigen::TensorBase<TensorType> >::registration() {
  bp::converter::registry::push_back(
      reinterpret_cast<void *(*)(_object *)>(&eigen_from_py_impl::convertible),
      &eigen_from_py_impl::construct, bp::type_id<TensorType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                          ,
      &eigenpy::expected_pytype_for_arg<TensorType>::get_pytype
#endif
  );
}

template <typename TensorType>
struct eigen_from_py_converter_impl<TensorType,
                                    Eigen::TensorBase<TensorType> > {
  static void registration() {
    EigenFromPy<TensorType>::registration();

    // Add conversion to Eigen::TensorBase<TensorType>
    typedef Eigen::TensorBase<TensorType> TensorBase;
    EigenFromPy<TensorBase>::registration();

    // Add conversion to Eigen::TensorRef<TensorType>
    typedef Eigen::TensorRef<TensorType> RefType;
    EigenFromPy<RefType>::registration();

    // Add conversion to Eigen::TensorRef<const TensorType>
    typedef const Eigen::TensorRef<const TensorType> ConstRefType;
    EigenFromPy<ConstRefType>::registration();
  }
};

template <typename TensorType>
struct EigenFromPy<Eigen::TensorBase<TensorType> > : EigenFromPy<TensorType> {
  typedef EigenFromPy<TensorType> EigenFromPyDerived;
  typedef Eigen::TensorBase<TensorType> Base;

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &EigenFromPy::construct, bp::type_id<Base>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                     ,
        &eigenpy::expected_pytype_for_arg<TensorType>::get_pytype
#endif
    );
  }
};

template <typename TensorType>
struct EigenFromPy<Eigen::TensorRef<TensorType> > {
  typedef Eigen::TensorRef<TensorType> RefType;
  typedef typename TensorType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj) {
    if (!call_PyArray_Check(pyObj)) return 0;
    PyArrayObject *pyArray = reinterpret_cast<PyArrayObject *>(pyObj);
    if (!PyArray_ISWRITEABLE(pyArray)) return 0;
    return EigenFromPy<TensorType>::convertible(pyObj);
  }

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &eigen_from_py_construct<RefType>, bp::type_id<RefType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                               ,
        &eigenpy::expected_pytype_for_arg<TensorType>::get_pytype
#endif
    );
  }
};

template <typename TensorType>
struct EigenFromPy<const Eigen::TensorRef<const TensorType> > {
  typedef const Eigen::TensorRef<const TensorType> ConstRefType;
  typedef typename TensorType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj) {
    return EigenFromPy<TensorType>::convertible(pyObj);
  }

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &eigen_from_py_construct<ConstRefType>, bp::type_id<ConstRefType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                                    ,
        &eigenpy::expected_pytype_for_arg<TensorType>::get_pytype
#endif
    );
  }
};

}  // namespace eigenpy

#endif  // __eigenpy_tensor_eigen_from_python_hpp__
