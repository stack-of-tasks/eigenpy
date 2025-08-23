//
// Copyright (c) 2024-2025 INRIA
//

#ifndef __eigenpy_sparse_eigen_from_python_hpp__
#define __eigenpy_sparse_eigen_from_python_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/scipy-type.hpp"
#include "eigenpy/scalar-conversion.hpp"

namespace eigenpy {

template <typename SparseMatrixType>
struct expected_pytype_for_arg<SparseMatrixType,
                               Eigen::SparseMatrixBase<SparseMatrixType>> {
  static PyTypeObject const *get_pytype() {
    PyTypeObject const *py_type = ScipyType::get_pytype<SparseMatrixType>();
    return py_type;
  }
};

}  // namespace eigenpy

namespace boost {
namespace python {
namespace converter {

template <typename Scalar, int Options, typename StorageIndex>
struct expected_pytype_for_arg<
    Eigen::SparseMatrix<Scalar, Options, StorageIndex>>
    : eigenpy::expected_pytype_for_arg<
          Eigen::SparseMatrix<Scalar, Options, StorageIndex>> {};

template <typename Scalar, int Options, typename StorageIndex>
struct rvalue_from_python_data<
    Eigen::SparseMatrix<Scalar, Options, StorageIndex> const &>
    : ::eigenpy::rvalue_from_python_data<
          Eigen::SparseMatrix<Scalar, Options, StorageIndex> const &> {
  typedef Eigen::SparseMatrix<Scalar, Options, StorageIndex> T;
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(T const &)
};

template <typename Derived>
struct rvalue_from_python_data<Eigen::SparseMatrixBase<Derived> const &>
    : ::eigenpy::rvalue_from_python_data<Derived const &> {
  EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
};

}  // namespace converter
}  // namespace python
}  // namespace boost

namespace boost {
namespace python {
namespace detail {
// template <typename TensorType>
// struct referent_storage<Eigen::TensorRef<TensorType> &> {
//   typedef Eigen::TensorRef<TensorType> RefType;
//   typedef ::eigenpy::details::referent_storage_eigen_ref<RefType>
//   StorageType; typedef typename ::eigenpy::aligned_storage<
//       referent_size<StorageType &>::value>::type type;
// };

// template <typename TensorType>
// struct referent_storage<const Eigen::TensorRef<const TensorType> &> {
//   typedef Eigen::TensorRef<const TensorType> RefType;
//   typedef ::eigenpy::details::referent_storage_eigen_ref<RefType>
//   StorageType; typedef typename ::eigenpy::aligned_storage<
//       referent_size<StorageType &>::value>::type type;
// };
}  // namespace detail
}  // namespace python
}  // namespace boost

namespace eigenpy {

template <typename SparseMatrixType>
struct eigen_from_py_impl<SparseMatrixType,
                          Eigen::SparseMatrixBase<SparseMatrixType>> {
  typedef typename SparseMatrixType::Scalar Scalar;

  /// \brief Determine if pyObj can be converted into a MatType object
  static void *convertible(PyObject *pyObj);

  /// \brief Allocate memory and copy pyObj in the new storage
  static void construct(PyObject *pyObj,
                        bp::converter::rvalue_from_python_stage1_data *memory);

  static void registration();
};

template <typename SparseMatrixType>
void *eigen_from_py_impl<
    SparseMatrixType,
    Eigen::SparseMatrixBase<SparseMatrixType>>::convertible(PyObject *pyObj) {
  const PyTypeObject *type = Py_TYPE(pyObj);
  const PyTypeObject *sparse_matrix_py_type =
      ScipyType::get_pytype<SparseMatrixType>();
  typedef typename SparseMatrixType::Scalar Scalar;

  if (type != sparse_matrix_py_type) return 0;

  bp::object obj(bp::handle<>(bp::borrowed(pyObj)));

  const int type_num = ScipyType::get_numpy_type_num(obj);

  if (!np_type_is_convertible_into_scalar<Scalar>(type_num)) return 0;

  return pyObj;
}

template <typename MatOrRefType>
void eigen_sparse_matrix_from_py_construct(
    PyObject *pyObj, bp::converter::rvalue_from_python_stage1_data *memory) {
  typedef typename MatOrRefType::Scalar Scalar;
  typedef typename MatOrRefType::StorageIndex StorageIndex;

  typedef Eigen::Map<MatOrRefType> MapMatOrRefType;

  bp::converter::rvalue_from_python_storage<MatOrRefType> *storage =
      reinterpret_cast<
          bp::converter::rvalue_from_python_storage<MatOrRefType> *>(
          reinterpret_cast<void *>(memory));
  void *raw_ptr = storage->storage.bytes;

  bp::object obj(bp::handle<>(bp::borrowed(pyObj)));

  const int type_num_python_sparse_matrix = ScipyType::get_numpy_type_num(obj);
  const int type_num_eigen_sparse_matrix = Register::getTypeCode<Scalar>();

  if (type_num_eigen_sparse_matrix == type_num_python_sparse_matrix) {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DataVector;
    //    typedef const Eigen::Ref<const DataVector> RefDataVector;
    DataVector data = bp::extract<DataVector>(obj.attr("data"));
    bp::tuple shape = bp::extract<bp::tuple>(obj.attr("shape"));
    typedef Eigen::Matrix<StorageIndex, Eigen::Dynamic, 1> StorageIndexVector;
    //    typedef const Eigen::Ref<const StorageIndexVector>
    //    RefStorageIndexVector;
    StorageIndexVector indices =
        bp::extract<StorageIndexVector>(obj.attr("indices"));
    StorageIndexVector indptr =
        bp::extract<StorageIndexVector>(obj.attr("indptr"));

    const Eigen::Index m = bp::extract<Eigen::Index>(shape[0]),
                       n = bp::extract<Eigen::Index>(shape[1]),
                       nnz = bp::extract<Eigen::Index>(obj.attr("nnz"));

    // Handle the specific case of the null matrix
    Scalar *data_ptr = nullptr;
    StorageIndex *indices_ptr = nullptr;
    if (nnz > 0) {
      data_ptr = data.data();
      indices_ptr = indices.data();
    }
    MapMatOrRefType sparse_map(m, n, nnz, indptr.data(), indices_ptr, data_ptr);

#if EIGEN_VERSION_AT_LEAST(3, 4, 90)
    sparse_map.sortInnerIndices();
#endif

    new (raw_ptr) MatOrRefType(sparse_map);
  }

  memory->convertible = storage->storage.bytes;
}

template <typename SparseMatrixType>
void eigen_from_py_impl<SparseMatrixType,
                        Eigen::SparseMatrixBase<SparseMatrixType>>::
    construct(PyObject *pyObj,
              bp::converter::rvalue_from_python_stage1_data *memory) {
  eigen_sparse_matrix_from_py_construct<SparseMatrixType>(pyObj, memory);
}

template <typename SparseMatrixType>
void eigen_from_py_impl<
    SparseMatrixType,
    Eigen::SparseMatrixBase<SparseMatrixType>>::registration() {
  bp::converter::registry::push_back(
      reinterpret_cast<void *(*)(_object *)>(&eigen_from_py_impl::convertible),
      &eigen_from_py_impl::construct, bp::type_id<SparseMatrixType>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                          ,
      &eigenpy::expected_pytype_for_arg<SparseMatrixType>::get_pytype
#endif
  );
}

template <typename SparseMatrixType>
struct eigen_from_py_converter_impl<SparseMatrixType,
                                    Eigen::SparseMatrixBase<SparseMatrixType>> {
  static void registration() {
    EigenFromPy<SparseMatrixType>::registration();

    // Add conversion to Eigen::SparseMatrixBase<SparseMatrixType>
    typedef Eigen::SparseMatrixBase<SparseMatrixType> SparseMatrixBase;
    EigenFromPy<SparseMatrixBase>::registration();

    //    // Add conversion to Eigen::Ref<SparseMatrixType>
    //    typedef Eigen::Ref<SparseMatrixType> RefType;
    //    EigenFromPy<SparseMatrixType>::registration();
    //
    //    // Add conversion to Eigen::Ref<const SparseMatrixType>
    //    typedef const Eigen::Ref<const SparseMatrixType> ConstRefType;
    //    EigenFromPy<ConstRefType>::registration();
  }
};

template <typename SparseMatrixType>
struct EigenFromPy<Eigen::SparseMatrixBase<SparseMatrixType>>
    : EigenFromPy<SparseMatrixType> {
  typedef EigenFromPy<SparseMatrixType> EigenFromPyDerived;
  typedef Eigen::SparseMatrixBase<SparseMatrixType> Base;

  static void registration() {
    bp::converter::registry::push_back(
        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
        &EigenFromPy::construct, bp::type_id<Base>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
                                     ,
        &eigenpy::expected_pytype_for_arg<SparseMatrixType>::get_pytype
#endif
    );
  }
};
//
// template <typename TensorType>
// struct EigenFromPy<Eigen::TensorRef<TensorType> > {
//  typedef Eigen::TensorRef<TensorType> RefType;
//  typedef typename TensorType::Scalar Scalar;
//
//  /// \brief Determine if pyObj can be converted into a MatType object
//  static void *convertible(PyObject *pyObj) {
//    if (!call_PyArray_Check(pyObj)) return 0;
//    PyArrayObject *pyArray = reinterpret_cast<PyArrayObject *>(pyObj);
//    if (!PyArray_ISWRITEABLE(pyArray)) return 0;
//    return EigenFromPy<TensorType>::convertible(pyObj);
//  }
//
//  static void registration() {
//    bp::converter::registry::push_back(
//        reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
//        &eigen_from_py_construct<RefType>, bp::type_id<RefType>()
// #ifndef BOOST_PYTHON_NO_PY_SIGNATURES
//                                               ,
//        &eigenpy::expected_pytype_for_arg<TensorType>::get_pytype
// #endif
//    );
//  }
//};

// template <typename TensorType>
// struct EigenFromPy<const Eigen::TensorRef<const TensorType> > {
//   typedef const Eigen::TensorRef<const TensorType> ConstRefType;
//   typedef typename TensorType::Scalar Scalar;
//
//   /// \brief Determine if pyObj can be converted into a MatType object
//   static void *convertible(PyObject *pyObj) {
//     return EigenFromPy<TensorType>::convertible(pyObj);
//   }
//
//   static void registration() {
//     bp::converter::registry::push_back(
//         reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
//         &eigen_from_py_construct<ConstRefType>, bp::type_id<ConstRefType>()
// #ifndef BOOST_PYTHON_NO_PY_SIGNATURES
//                                                     ,
//         &eigenpy::expected_pytype_for_arg<TensorType>::get_pytype
// #endif
//     );
//   }
// };

}  // namespace eigenpy

#endif  // __eigenpy_sparse_eigen_from_python_hpp__
