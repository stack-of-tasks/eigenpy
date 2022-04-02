//
// Copyright (c) 2014-2020 CNRS INRIA
//

#ifndef __eigenpy_scalar_conversion_hpp__
#define __eigenpy_scalar_conversion_hpp__

#include "eigenpy/config.hpp"

namespace eigenpy {
template <typename SCALAR1, typename SCALAR2>
struct FromTypeToType : public boost::false_type {};

template <typename SCALAR>
struct FromTypeToType<SCALAR, SCALAR> : public boost::true_type {};

template <>
struct FromTypeToType<int, long> : public boost::true_type {};
template <>
struct FromTypeToType<int, float> : public boost::true_type {};
template <>
struct FromTypeToType<int, std::complex<float> > : public boost::true_type {};
template <>
struct FromTypeToType<int, double> : public boost::true_type {};
template <>
struct FromTypeToType<int, std::complex<double> > : public boost::true_type {};
template <>
struct FromTypeToType<int, long double> : public boost::true_type {};
template <>
struct FromTypeToType<int, std::complex<long double> >
    : public boost::true_type {};

template <>
struct FromTypeToType<long, float> : public boost::true_type {};
template <>
struct FromTypeToType<long, std::complex<float> > : public boost::true_type {};
template <>
struct FromTypeToType<long, double> : public boost::true_type {};
template <>
struct FromTypeToType<long, std::complex<double> > : public boost::true_type {};
template <>
struct FromTypeToType<long, long double> : public boost::true_type {};
template <>
struct FromTypeToType<long, std::complex<long double> >
    : public boost::true_type {};

template <>
struct FromTypeToType<float, std::complex<float> > : public boost::true_type {};
template <>
struct FromTypeToType<float, double> : public boost::true_type {};
template <>
struct FromTypeToType<float, std::complex<double> > : public boost::true_type {
};
template <>
struct FromTypeToType<float, long double> : public boost::true_type {};
template <>
struct FromTypeToType<float, std::complex<long double> >
    : public boost::true_type {};

template <>
struct FromTypeToType<double, std::complex<double> > : public boost::true_type {
};
template <>
struct FromTypeToType<double, long double> : public boost::true_type {};
template <>
struct FromTypeToType<double, std::complex<long double> >
    : public boost::true_type {};
}  // namespace eigenpy

#endif  // __eigenpy_scalar_conversion_hpp__
