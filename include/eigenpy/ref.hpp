/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_ref_hpp__
#define __eigenpy_ref_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/stride.hpp"

// For old Eigen versions, EIGEN_DEVICE_FUNC is not defined.
// We must define it just in the scope of this file.
#if !EIGEN_VERSION_AT_LEAST(3,2,90)
#define EIGEN_DEVICE_FUNC
#endif

namespace eigenpy
{

  template<typename PlainObjectTypeT>
  struct Ref : Eigen::Ref<PlainObjectTypeT,EIGENPY_DEFAULT_ALIGNMENT_VALUE,typename StrideType<PlainObjectTypeT>::type>
  {
  public:
    typedef Eigen::Ref<PlainObjectTypeT,EIGENPY_DEFAULT_ALIGNMENT_VALUE,typename eigenpy::template StrideType<PlainObjectTypeT>::type> Base;

  private:
    typedef Eigen::internal::traits<Base> Traits;
    template<typename Derived>
    EIGEN_DEVICE_FUNC inline Ref(const Eigen::PlainObjectBase<Derived>& expr,
                                 typename Eigen::internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0);
    
  public:
    
    typedef typename Eigen::internal::traits<Base>::Scalar Scalar; /*!< \brief Numeric type, e.g. float, double, int or std::complex<float>. */ \
    typedef typename Eigen::NumTraits<Scalar>::Real RealScalar; /*!< \brief The underlying numeric type for composed scalar types. \details In cases where Scalar is e.g. std::complex<T>, T were corresponding to RealScalar. */ \
    typedef typename Base::CoeffReturnType CoeffReturnType; /*!< \brief The return type for coefficient access. \details Depending on whether the object allows direct coefficient access (e.g. for a MatrixXd), this type is either 'const Scalar&' or simply 'Scalar' for objects that do not allow direct coefficient access. */
    typedef typename Eigen::internal::ref_selector<Base>::type Nested;
    typedef typename Eigen::internal::traits<Base>::StorageKind StorageKind;
#if EIGEN_VERSION_AT_LEAST(3,2,90)
    typedef typename Eigen::internal::traits<Base>::StorageIndex StorageIndex;
#else
    typedef typename Eigen::internal::traits<Base>::Index StorageIndex;
#endif
    enum { RowsAtCompileTime = Eigen::internal::traits<Base>::RowsAtCompileTime,
      ColsAtCompileTime = Eigen::internal::traits<Base>::ColsAtCompileTime,
      Flags = Eigen::internal::traits<Base>::Flags,
      SizeAtCompileTime = Base::SizeAtCompileTime,
      MaxSizeAtCompileTime = Base::MaxSizeAtCompileTime,
      IsVectorAtCompileTime = Base::IsVectorAtCompileTime };
    using Base::derived;
    using Base::const_cast_derived;
    typedef typename Base::PacketScalar PacketScalar;
    
    template<typename Derived>
    EIGEN_DEVICE_FUNC inline Ref(Eigen::PlainObjectBase<Derived>& expr,
                                 typename Eigen::internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0)
    : Base(expr.derived())
    {}
    
    template<typename Derived>
    EIGEN_DEVICE_FUNC inline Ref(const Eigen::DenseBase<Derived>& expr,
                                 typename Eigen::internal::enable_if<bool(Traits::template match<Derived>::MatchAtCompileTime),Derived>::type* = 0)
    : Base(expr.derived())
    {}
    
#if EIGEN_COMP_MSVC_STRICT && (EIGEN_COMP_MSVC < 1900 ||  defined(__CUDACC_VER__)) // for older MSVC versions, as well as 1900 && CUDA 8, using the base operator is sufficient (cf Bugs 1000, 1324)
    using Base::operator =;
#elif EIGEN_COMP_CLANG // workaround clang bug (see http://forum.kde.org/viewtopic.php?f=74&t=102653)
    using Base::operator =; \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Ref& operator=(const Ref& other) { Base::operator=(other); return *this; } \
    template <typename OtherDerived> \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Ref& operator=(const Eigen::DenseBase<OtherDerived>& other) { Base::operator=(other.derived()); return *this; }
#else
    using Base::operator =; \
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Ref& operator=(const Ref& other) \
    { \
      Base::operator=(other); \
      return *this; \
    }
#endif
    
  }; // struct Ref<PlainObjectType>
}

#if !EIGEN_VERSION_AT_LEAST(3,2,90)
#undef EIGEN_DEVICE_FUNC
#endif

#endif // ifndef __eigenpy_ref_hpp__
