//
// Copyright (C) 2020 INRIA
// Copyright (C) 2024 LAAS-CNRS, INRIA
//
#ifndef __eigenpy_deprecation_hpp__
#define __eigenpy_deprecation_hpp__

#include "eigenpy/fwd.hpp"

namespace eigenpy {

enum class DeprecationType { DEPRECATION, FUTURE };

namespace detail {

inline PyObject *deprecationTypeToPyObj(DeprecationType dep) {
  switch (dep) {
    case DeprecationType::DEPRECATION:
      return PyExc_DeprecationWarning;
    case DeprecationType::FUTURE:
      return PyExc_FutureWarning;
    default:  // The switch handles all cases explicitly, this should never be
              // triggered.
      throw std::invalid_argument(
          "Undefined DeprecationType - this should never be triggered.");
  }
}

}  // namespace detail

/// @brief A Boost.Python call policy which triggers a Python warning on
/// precall.
template <DeprecationType deprecation_type = DeprecationType::DEPRECATION,
          class BasePolicy = bp::default_call_policies>
struct deprecation_warning_policy : BasePolicy {
  using result_converter = typename BasePolicy::result_converter;
  using argument_package = typename BasePolicy::argument_package;

  deprecation_warning_policy(const std::string &warning_msg)
      : BasePolicy(), m_what(warning_msg) {}

  std::string what() const { return m_what; }

  const BasePolicy *derived() const {
    return static_cast<const BasePolicy *>(this);
  }

  template <class ArgPackage>
  bool precall(const ArgPackage &args) const {
    PyErr_WarnEx(detail::deprecationTypeToPyObj(deprecation_type),
                 m_what.c_str(), 1);
    return derived()->precall(args);
  }

 protected:
  const std::string m_what;
};

template <DeprecationType deprecation_type = DeprecationType::DEPRECATION,
          class BasePolicy = bp::default_call_policies>
struct deprecated_function
    : deprecation_warning_policy<deprecation_type, BasePolicy> {
  deprecated_function(const std::string &msg =
                          "This function has been marked as deprecated, and "
                          "will be removed in the future.")
      : deprecation_warning_policy<deprecation_type, BasePolicy>(msg) {}
};

template <DeprecationType deprecation_type = DeprecationType::DEPRECATION,
          class BasePolicy = bp::default_call_policies>
struct deprecated_member
    : deprecation_warning_policy<deprecation_type, BasePolicy> {
  deprecated_member(const std::string &msg =
                        "This attribute or method has been marked as "
                        "deprecated, and will be removed in the future.")
      : deprecation_warning_policy<deprecation_type, BasePolicy>(msg) {}
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_deprecation_hpp__
