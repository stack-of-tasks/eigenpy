//
// Copyright (c) 2026 INRIA
//

#ifndef __eigenpy_user_type_traits_hpp__
#define __eigenpy_user_type_traits_hpp__

#include <cstring>

namespace eigenpy {

///
/// \brief Customization point for user-defined numpy scalar types whose
///        all-zero byte pattern is *not* a valid value.
///
/// eigenpy registers user types with NPY_NEEDS_INIT: numpy zero-fills fresh
/// buffers (np.zeros, np.empty, ufunc outputs, ...) but never runs a C++
/// constructor on the elements. The dtype slot functions and ufunc loops may
/// therefore encounter slots that were never constructed. For most scalar
/// types the all-zero pattern is simply the value 0 and nothing special is
/// needed. For pointer-backed scalars (e.g. Boost.Multiprecision mpfr/mpc
/// numbers, whose all-zero state holds a null limb pointer) reading such a
/// slot crashes inside the backend library.
///
/// Specialize this trait for such types so that eigenpy treats a
/// never-constructed slot as an exact T(0) on read, and zeroes the
/// destination bytes before assignment on write.
///
/// Requirement on specializing types: T must tolerate assignment through
/// T::operator= into all-zero storage (Boost.Multiprecision number types
/// satisfy this; it is already implied by registering the type with
/// NPY_NEEDS_INIT).
///
template <typename T>
struct user_type_traits {
  /// \brief Compile-time switch. When false (the default) every guard
  ///        compiles out and behavior is identical to previous eigenpy
  ///        releases.
  enum { requires_initialization = false };

  /// \brief Detect the never-constructed pattern. Only ever called when
  ///        requires_initialization is true.
  static bool is_uninitialized(const T& /*x*/) { return false; }
};

namespace internal {

/// \brief Read-side guard: yields \p zero for a never-constructed slot,
///        \p x otherwise.
template <typename T>
inline const T& value_or_zero(const T& x, const T& zero) {
  if (user_type_traits<T>::requires_initialization &&
      user_type_traits<T>::is_uninitialized(x))
    return zero;
  return x;
}

/// \brief Write-side guard: force the destination back onto T::operator='s
///        init-before-set path even if the allocator delivered dirty
///        (non-zero garbage) memory.
template <typename T>
inline void prepare_destination(void* dest) {
  if (user_type_traits<T>::requires_initialization)
    std::memset(dest, 0, sizeof(T));
}

}  // namespace internal
}  // namespace eigenpy

#endif  // ifndef __eigenpy_user_type_traits_hpp__
