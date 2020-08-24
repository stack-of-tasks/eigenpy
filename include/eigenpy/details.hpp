/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#ifndef __eigenpy_details_hpp__
#define __eigenpy_details_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/eigenpy.hpp"

#include "eigenpy/numpy-type.hpp"
#include "eigenpy/scalar-conversion.hpp"

#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigen-from-python.hpp"

#include "eigenpy/registration.hpp"
#include "eigenpy/exception.hpp"

namespace boost { namespace python { namespace detail {

  template<class MatType>
  struct referent_size<Eigen::MatrixBase<MatType>&>
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::MatrixBase<MatType> >
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::EigenBase<MatType>&>
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::EigenBase<MatType> >
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::PlainObjectBase<MatType>&>
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::PlainObjectBase<MatType> >
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

}}}

namespace eigenpy
{
  template<typename MatType,typename EigenEquivalentType>
  EIGENPY_DEPRECATED
  void enableEigenPySpecific()
  {
    enableEigenPySpecific<MatType>();
  }

  template<typename MatType>
  void enableEigenPySpecific()
  {
    if(check_registration<MatType>()) return;
    
    // to-python
    EigenToPyConverter<MatType>::registration();
#if EIGEN_VERSION_AT_LEAST(3,2,0)
    EigenToPyConverter< Eigen::Ref<MatType> >::registration();
#endif
    
    // from-python
    EigenFromPyConverter<MatType>::registration();
  }

} // namespace eigenpy

#endif // ifndef __eigenpy_details_hpp__
