/*
 * Copyright 2014-2023 CNRS INRIA
 */

#ifndef __eigenpy_fwd_hpp__
#define __eigenpy_fwd_hpp__

#if defined(__clang__)
#define EIGENPY_CLANG_COMPILER
#elif defined(__GNUC__)
#define EIGENPY_GCC_COMPILER
#elif defined(_MSC_VER)
#define EIGENPY_MSVC_COMPILER
#endif

#if (__cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703))
#define EIGENPY_WITH_CXX17_SUPPORT
#endif

#if (__cplusplus >= 201402L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201403))
#define EIGENPY_WITH_CXX14_SUPPORT
#endif

#if (__cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1600))
#define EIGENPY_WITH_CXX11_SUPPORT
#endif

#define EIGENPY_STRING_LITERAL(string) #string
#define EIGENPY_STRINGIZE(string) EIGENPY_STRING_LITERAL(string)
#define _EIGENPY_PPCAT(A, B) A##B
#define EIGENPY_PPCAT(A, B) _EIGENPY_PPCAT(A, B)
#define EIGENPY_STRINGCAT(A, B) A B

// For more details, visit
// https://stackoverflow.com/questions/171435/portability-of-warning-preprocessor-directive
#if defined(EIGENPY_CLANG_COMPILER) || defined(EIGENPY_GCC_COMPILER)
#define EIGENPY_PRAGMA(x) _Pragma(#x)
#define EIGENPY_PRAGMA_MESSAGE(the_message) \
  EIGENPY_PRAGMA(GCC message the_message)
#define EIGENPY_PRAGMA_WARNING(the_message) \
  EIGENPY_PRAGMA(GCC warning the_message)
#define EIGENPY_PRAGMA_DEPRECATED(the_message) \
  EIGENPY_PRAGMA_WARNING(Deprecated : the_message)
#define EIGENPY_PRAGMA_DEPRECATED_HEADER(old_header, new_header) \
  EIGENPY_PRAGMA_WARNING(                                        \
      Deprecated header file                                     \
      : #old_header has been replaced                            \
            by #new_header.\n Please use #new_header instead of #old_header.)
#elif defined(WIN32)
#define EIGENPY_PRAGMA(x) __pragma(#x)
#define EIGENPY_PRAGMA_MESSAGE(the_message) \
  EIGENPY_PRAGMA(message(#the_message))
#define EIGENPY_PRAGMA_WARNING(the_message) \
  EIGENPY_PRAGMA(message(EIGENPY_STRINGCAT("WARNING: ", the_message)))
#endif

#define EIGENPY_DEPRECATED_MACRO(macro, the_message) \
  EIGENPY_PRAGMA_WARNING(                            \
      EIGENPY_STRINGCAT("this macro is deprecated: ", the_message))
#define EIGENPY_DEPRECATED_FILE(the_message) \
  EIGENPY_PRAGMA_WARNING(                    \
      EIGENPY_STRINGCAT("this file is deprecated: ", the_message))

#define EIGENPY_DOCUMENTATION_START_IGNORE  /// \cond
#define EIGENPY_DOCUMENTATION_END_IGNORE    /// \endcond

#include "eigenpy/config.hpp"

// Silence a warning about a deprecated use of boost bind by boost python
// at least fo boost 1.73 to 1.75
// ref. https://github.com/stack-of-tasks/tsid/issues/128
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/python.hpp>
#include <boost/python/scope.hpp>

namespace eigenpy {

namespace bp = boost::python;

}

#define NO_IMPORT_ARRAY
#include "eigenpy/numpy.hpp"
#undef NO_IMPORT_ARRAY

#undef BOOST_BIND_GLOBAL_PLACEHOLDERS

#include <Eigen/Core>
#include <Eigen/Geometry>

#ifdef EIGENPY_WITH_CXX11_SUPPORT
#include <unsupported/Eigen/CXX11/Tensor>
#define EIGENPY_WITH_TENSOR_SUPPORT
#endif

#if EIGEN_VERSION_AT_LEAST(3, 2, 90)
#define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Aligned16
#else
#define EIGENPY_DEFAULT_ALIGNMENT_VALUE Eigen::Aligned
#endif

#define EIGENPY_DEFAULT_ALIGN_BYTES EIGEN_DEFAULT_ALIGN_BYTES

#define EIGENPY_NO_ALIGNMENT_VALUE Eigen::Unaligned

#define EIGENPY_UNUSED_VARIABLE(var) (void)(var)
#define EIGENPY_UNUSED_TYPE(type) EIGENPY_UNUSED_VARIABLE((type *)(NULL))

#ifdef EIGENPY_WITH_CXX11_SUPPORT
#include <memory>
#define EIGENPY_SHARED_PTR_HOLDER_TYPE(T) ::std::shared_ptr<T>
#else
#include <boost/shared_ptr.hpp>
#define EIGENPY_SHARED_PTR_HOLDER_TYPE(T) ::boost::shared_ptr<T>
#endif

namespace eigenpy {
template <typename MatType,
          typename Scalar =
              typename boost::remove_reference<MatType>::type::Scalar>
struct EigenToPy;
template <typename MatType,
          typename Scalar =
              typename boost::remove_reference<MatType>::type::Scalar>
struct EigenFromPy;

template <typename T>
struct remove_const_reference {
  typedef typename boost::remove_const<
      typename boost::remove_reference<T>::type>::type type;
};

template <typename EigenType>
struct get_eigen_base_type {
  typedef typename remove_const_reference<EigenType>::type EigenType_;
  typedef typename boost::mpl::if_<
      boost::is_base_of<Eigen::MatrixBase<EigenType_>, EigenType_>,
      Eigen::MatrixBase<EigenType_>
#ifdef EIGENPY_WITH_TENSOR_SUPPORT
      ,
      typename boost::mpl::if_<
          boost::is_base_of<Eigen::TensorBase<EigenType_>, EigenType_>,
          Eigen::TensorBase<EigenType_>, void>::type
#else
      ,
      void
#endif
      >::type _type;

  typedef typename boost::mpl::if_<
      boost::is_const<typename boost::remove_reference<EigenType>::type>,
      const _type, _type>::type type;
};

template <typename EigenType>
struct get_eigen_ref_plain_type;

template <typename MatType, int Options, typename Stride>
struct get_eigen_ref_plain_type<Eigen::Ref<MatType, Options, Stride> > {
  typedef typename Eigen::internal::traits<
      Eigen::Ref<MatType, Options, Stride> >::PlainObjectType type;
};

#ifdef EIGENPY_WITH_TENSOR_SUPPORT
template <typename TensorType>
struct get_eigen_ref_plain_type<Eigen::TensorRef<TensorType> > {
  typedef TensorType type;
};
#endif
}  // namespace eigenpy

#include "eigenpy/alignment.hpp"

#endif  // ifndef __eigenpy_fwd_hpp__
