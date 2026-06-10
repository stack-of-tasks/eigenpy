//
// Copyright (c) 2024 INRIA
//
//

#ifndef __eigenpy_utils_traits_hpp__
#define __eigenpy_utils_traits_hpp__

#include <type_traits>
#include <string>
#include <complex>

namespace eigenpy {

namespace details {

/// Trait to remove const&
template <typename T>
struct remove_cvref : std::remove_cv<typename std::remove_reference<T>::type> {
};

/// Trait to detect if T is a class or an union
template <typename T>
struct is_class_or_union
    : std::integral_constant<bool, std::is_class<T>::value ||
                                       std::is_union<T>::value> {};

/// trait to detect if T is a std::complex managed by Boost Python
template <typename T>
struct is_python_complex : std::false_type {};

/// From boost/python/converter/builtin_converters
template <>
struct is_python_complex<std::complex<float>> : std::true_type {};
template <>
struct is_python_complex<std::complex<double>> : std::true_type {};
template <>
struct is_python_complex<std::complex<long double>> : std::true_type {};

template <typename T>
struct is_python_primitive_type_helper
    : std::integral_constant<bool, !is_class_or_union<T>::value ||
                                       std::is_same<T, std::string>::value ||
                                       std::is_same<T, std::wstring>::value ||
                                       is_python_complex<T>::value> {};

/// Trait to detect if T is a Python primitive type
template <typename T>
struct is_python_primitive_type
    : is_python_primitive_type_helper<typename remove_cvref<T>::type> {};

}  // namespace details

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_traits_hpp__
