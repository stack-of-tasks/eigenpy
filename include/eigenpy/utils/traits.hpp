//
// Copyright (c) 2024 INRIA
//
//
#include <type_traits>

#ifndef __eigenpy_utils_traits_hpp__
#define __eigenpy_utils_traits_hpp__

namespace eigenpy {

namespace details {

/// Trait to detect if T is a class or an union
template <typename T>
struct is_class_or_union
    : std::integral_constant<bool, std::is_class<T>::value ||
                                       std::is_union<T>::value> {};

template <typename T>
struct remove_cvref : std::remove_cv<typename std::remove_reference<T>::type> {
};

/// Trait to remove cvref and call is_class_or_union
template <typename T>
struct is_class_or_union_remove_cvref
    : is_class_or_union<typename remove_cvref<T>::type> {};

}  // namespace details

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_traits_hpp__
