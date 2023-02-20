/*
 * Copyright 2023 INRIA
 */

#ifndef __eigenpy_alignment_hpp__
#define __eigenpy_alignment_hpp__

#include <boost/python/detail/referent_storage.hpp>
#include <boost/python/converter/arg_from_python.hpp>
#include <boost/python/converter/rvalue_from_python_data.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <eigenpy/utils/is-aligned.hpp>

namespace eigenpy {

template <std::size_t size, std::size_t alignment = EIGENPY_DEFAULT_ALIGN_BYTES>
struct aligned_storage {
  union type {
    typename ::boost::aligned_storage<size, alignment>::type data;
    char bytes[size];
  };
};

template <class Data>
struct aligned_instance {
  PyObject_VAR_HEAD PyObject *dict;
  PyObject *weakrefs;
  boost::python::instance_holder *objects;

  typename aligned_storage<sizeof(Data)>::type storage;
};

inline void *aligned_malloc(
    std::size_t size, std::size_t alignment = EIGENPY_DEFAULT_ALIGN_BYTES) {
  void *original = std::malloc(size + alignment);
  if (original == 0) return 0;
  if (is_aligned(original, alignment)) return original;
  void *aligned =
      reinterpret_cast<void *>((reinterpret_cast<std::size_t>(original) &
                                ~(std::size_t(alignment - 1))) +
                               alignment);
  *(reinterpret_cast<void **>(aligned) - 1) = original;
  return aligned;
}

}  // namespace eigenpy

namespace boost {
namespace python {
namespace detail {

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct referent_storage<
    Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &> {
  typedef Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> T;
  typedef
      typename eigenpy::aligned_storage<referent_size<T &>::value>::type type;
};

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
struct referent_storage<
    const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &> {
  typedef Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> T;
  typedef
      typename eigenpy::aligned_storage<referent_size<T &>::value>::type type;
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename Scalar, int Rank, int Options, typename IndexType>
struct referent_storage<Eigen::Tensor<Scalar, Rank, Options, IndexType> &> {
  typedef Eigen::Tensor<Scalar, Rank, Options, IndexType> T;
  typedef
      typename eigenpy::aligned_storage<referent_size<T &>::value>::type type;
};

template <typename Scalar, int Rank, int Options, typename IndexType>
struct referent_storage<
    const Eigen::Tensor<Scalar, Rank, Options, IndexType> &> {
  typedef Eigen::Tensor<Scalar, Rank, Options, IndexType> T;
  typedef
      typename eigenpy::aligned_storage<referent_size<T &>::value>::type type;
};
#endif

template <typename Scalar, int Options>
struct referent_storage<Eigen::Quaternion<Scalar, Options> &> {
  typedef Eigen::Quaternion<Scalar, Options> T;
  typedef
      typename eigenpy::aligned_storage<referent_size<T &>::value>::type type;
};

template <typename Scalar, int Options>
struct referent_storage<const Eigen::Quaternion<Scalar, Options> &> {
  typedef Eigen::Quaternion<Scalar, Options> T;
  typedef
      typename eigenpy::aligned_storage<referent_size<T &>::value>::type type;
};

}  // namespace detail
}  // namespace python
}  // namespace boost

namespace boost {
namespace python {
namespace objects {

// Force alignment of instance with value_holder
template <typename Derived>
struct instance<value_holder<Derived> >
    : ::eigenpy::aligned_instance<value_holder<Derived> > {};

}  // namespace objects
}  // namespace python
}  // namespace boost

namespace eigenpy {

template <class T>
struct call_destructor {
  static void run(void *bytes) {
    typedef typename boost::remove_const<
        typename boost::remove_reference<T>::type>::type T_;
    static_cast<T_ *>((void *)bytes)->~T_();
  }
};

template <class T>
struct rvalue_from_python_data
    : ::boost::python::converter::rvalue_from_python_storage<T> {
#if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) &&                        \
    (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) &&                 \
    (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) &&                   \
    !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing \
    this */
  // This must always be a POD struct with m_data its first member.
  BOOST_STATIC_ASSERT(
      BOOST_PYTHON_OFFSETOF(
          ::boost::python::converter::rvalue_from_python_storage<T>, stage1) ==
      0);
#endif

  // The usual constructor
  rvalue_from_python_data(
      ::boost::python::converter::rvalue_from_python_stage1_data const
          &_stage1) {
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
    if (this->stage1.convertible == this->storage.bytes) {
      void *storage = reinterpret_cast<void *>(this->storage.bytes);
      call_destructor<T>::run(storage);
    }
  }
};

}  // namespace eigenpy

#endif  // __eigenpy_alignment_hpp__
