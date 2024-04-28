//
// Copyright (C) 2024 LAAS-CNRS, INRIA
//
#ifndef __eigenpy_deprecation_hpp__
#define __eigenpy_deprecation_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {

enum class DeprecationType { DEPRECATION, FUTURE };

namespace detail {

constexpr PyObject *deprecationTypeToPyObj(DeprecationType dep) {
  switch (dep) {
    case DeprecationType::DEPRECATION:
      return PyExc_DeprecationWarning;
    case DeprecationType::FUTURE:
      return PyExc_FutureWarning;
  }
}

}  // namespace detail

constexpr char defaultDeprecationMessage[] =
    "This function or attribute has been marked as deprecated, and will be "
    "removed in the "
    "future.";

/// @brief A Boost.Python call policy which triggers a Python warning on
/// precall.
template <DeprecationType deprecation_type = DeprecationType::DEPRECATION,
          class BasePolicy = bp::default_call_policies>
struct deprecation_warning_policy : BasePolicy {
  using result_converter = typename BasePolicy::result_converter;
  using argument_package = typename BasePolicy::argument_package;

  deprecation_warning_policy(
      const std::string &warning_msg = defaultDeprecationMessage)
      : BasePolicy(), m_what(warning_msg) {}

  const std::string what() const { return m_what; }

  const BasePolicy *derived() const {
    return static_cast<const BasePolicy *>(this);
  }

  template <class ArgPackage>
  bool precall(const ArgPackage &args) const {
    PyErr_WarnEx(detail::deprecationTypeToPyObj(deprecation_type),
                 m_what.c_str(), 1);
    return derived()->precall(args);
  }

 private:
  const std::string m_what;
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_deprecation_hpp__
