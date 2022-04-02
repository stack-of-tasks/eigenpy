/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_utils_scalar_name_hpp__
#define __eigenpy_utils_scalar_name_hpp__

#include <complex>
#include <string>

namespace eigenpy {
template <typename Scalar>
struct scalar_name {
  static std::string shortname();
};

template <>
struct scalar_name<float> {
  static std::string shortname() { return "f"; };
};

template <>
struct scalar_name<double> {
  static std::string shortname() { return "d"; };
};

template <>
struct scalar_name<long double> {
  static std::string shortname() { return "ld"; };
};

template <typename Scalar>
struct scalar_name<std::complex<Scalar> > {
  static std::string shortname() { return "c" + scalar_name<Scalar>(); };
};
}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_scalar_name_hpp__
