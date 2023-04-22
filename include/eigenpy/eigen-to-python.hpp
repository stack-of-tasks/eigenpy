//
// Copyright (c) 2014-2023 CNRS INRIA
//

#ifndef __eigenpy_eigen_to_python_hpp__
#define __eigenpy_eigen_to_python_hpp__

#include <boost/type_traits.hpp>

#include "eigenpy/fwd.hpp"

#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/numpy-allocator.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/registration.hpp"

namespace boost {
namespace python {

template <typename MatrixRef, class MakeHolder>
struct to_python_indirect_eigen {
  template <class U>
  inline PyObject* operator()(U const& mat) const {
    return eigenpy::EigenToPy<MatrixRef>::convert(const_cast<U&>(mat));
  }

#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
  inline PyTypeObject const* get_pytype() const {
    return converter::registered_pytype<MatrixRef>::get_pytype();
  }
#endif
};

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime,
          int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime,
          class MakeHolder>
struct to_python_indirect<
    Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                  MaxRowsAtCompileTime, MaxColsAtCompileTime>&,
    MakeHolder>
    : to_python_indirect_eigen<
          Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                        MaxRowsAtCompileTime, MaxColsAtCompileTime>&,
          MakeHolder> {};

template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime,
          int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime,
          class MakeHolder>
struct to_python_indirect<
    const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime, Options,
                        MaxRowsAtCompileTime, MaxColsAtCompileTime>&,
    MakeHolder>
    : to_python_indirect_eigen<
          const Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime,
                              Options, MaxRowsAtCompileTime,
                              MaxColsAtCompileTime>&,
          MakeHolder> {};

}  // namespace python
}  // namespace boost

namespace eigenpy {

EIGENPY_DOCUMENTATION_START_IGNORE

template <typename EigenType,
          typename BaseType = typename get_eigen_base_type<EigenType>::type>
struct eigen_to_py_impl;

template <typename MatType>
struct eigen_to_py_impl_matrix;

template <typename MatType>
struct eigen_to_py_impl<MatType, Eigen::MatrixBase<MatType> >
    : eigen_to_py_impl_matrix<MatType> {};

template <typename MatType>
struct eigen_to_py_impl<MatType&, Eigen::MatrixBase<MatType> >
    : eigen_to_py_impl_matrix<MatType&> {};

template <typename MatType>
struct eigen_to_py_impl<const MatType, const Eigen::MatrixBase<MatType> >
    : eigen_to_py_impl_matrix<const MatType> {};

template <typename MatType>
struct eigen_to_py_impl<const MatType&, const Eigen::MatrixBase<MatType> >
    : eigen_to_py_impl_matrix<const MatType&> {};

template <typename MatType>
struct eigen_to_py_impl_matrix {
  static PyObject* convert(
      typename boost::add_reference<
          typename boost::add_const<MatType>::type>::type mat) {
    typedef typename boost::remove_const<
        typename boost::remove_reference<MatType>::type>::type MatrixDerived;

    assert((mat.rows() < INT_MAX) && (mat.cols() < INT_MAX) &&
           "Matrix range larger than int ... should never happen.");
    const npy_intp R = (npy_intp)mat.rows(), C = (npy_intp)mat.cols();

    PyArrayObject* pyArray;
    // Allocate Python memory
    if ((((!(C == 1) != !(R == 1)) && !MatrixDerived::IsVectorAtCompileTime) ||
         MatrixDerived::IsVectorAtCompileTime))  // Handle array with a single
                                                 // dimension
    {
      npy_intp shape[1] = {C == 1 ? R : C};
      pyArray = NumpyAllocator<MatType>::allocate(
          const_cast<MatrixDerived&>(mat.derived()), 1, shape);
    } else {
      npy_intp shape[2] = {R, C};
      pyArray = NumpyAllocator<MatType>::allocate(
          const_cast<MatrixDerived&>(mat.derived()), 2, shape);
    }

    // Create an instance (either np.array or np.matrix)
    return NumpyType::make(pyArray).ptr();
  }
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType>
struct eigen_to_py_impl_tensor;

template <typename TensorType>
struct eigen_to_py_impl<TensorType, Eigen::TensorBase<TensorType> >
    : eigen_to_py_impl_tensor<TensorType> {};

template <typename TensorType>
struct eigen_to_py_impl<const TensorType, const Eigen::TensorBase<TensorType> >
    : eigen_to_py_impl_tensor<const TensorType> {};

template <typename TensorType>
struct eigen_to_py_impl_tensor {
  static PyObject* convert(
      typename boost::add_reference<
          typename boost::add_const<TensorType>::type>::type tensor) {
    //    typedef typename boost::remove_const<
    //        typename boost::remove_reference<Tensor>::type>::type
    //        TensorDerived;

    static const int NumIndices = TensorType::NumIndices;
    npy_intp shape[NumIndices];
    for (int k = 0; k < NumIndices; ++k) shape[k] = tensor.dimension(k);

    PyArrayObject* pyArray = NumpyAllocator<TensorType>::allocate(
        const_cast<TensorType&>(tensor), NumIndices, shape);

    // Create an instance (either np.array or np.matrix)
    return NumpyType::make(pyArray).ptr();
  }
};
#endif

EIGENPY_DOCUMENTATION_END_IGNORE

#ifdef EIGENPY_MSVC_COMPILER
template <typename EigenType>
struct EigenToPy<EigenType,
                 typename boost::remove_reference<EigenType>::type::Scalar>
#else
template <typename EigenType, typename _Scalar>
struct EigenToPy
#endif
    : eigen_to_py_impl<EigenType> {
  static PyTypeObject const* get_pytype() { return getPyArrayType(); }
};

template <typename MatType>
struct EigenToPyConverter {
  static void registration() {
    bp::to_python_converter<MatType, EigenToPy<MatType>, true>();
  }
};
}  // namespace eigenpy

#endif  // __eigenpy_eigen_to_python_hpp__
