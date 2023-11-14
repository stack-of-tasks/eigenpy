/*
 * Copyright 2023, INRIA
 */

#ifndef __eigenpy_registration_class_hpp__
#define __eigenpy_registration_class_hpp__

#include <boost/python/class.hpp>

#include "eigenpy/fwd.hpp"

namespace eigenpy {

/*! Copy of the \see boost::python::class_
 * This class allow to add methods to an existing class without registering it
 * again.
 **/
template <class W>
class registration_class {
 public:
  using self = registration_class;

  /// \p object Hold the namespace of the class that will be modified
  registration_class(bp::object object) : m_object(object) {}

  /// \see boost::python::class_::def(bp::def_visitor<Derived> const& visitor)
  template <class Visitor>
  self& def(Visitor const& visitor) {
    visitor.visit(*this);
    return *this;
  }

  template <class DerivedVisitor>
  self& def(bp::def_visitor<DerivedVisitor> const& visitor) {
    static_cast<DerivedVisitor const&>(visitor).visit(*this);
    return *this;
  }

  /// \see boost::python::class_::def(char const* name, F f)
  template <class F>
  self& def(char const* name, F f) {
    def_impl(bp::detail::unwrap_wrapper((W*)0), name, f,
             bp::detail::def_helper<char const*>(0), &f);
    return *this;
  }

  /// \see boost::python::class_::def(char const* name, A1 a1, A2 const& a2)
  template <class A1, class A2>
  self& def(char const* name, A1 a1, A2 const& a2) {
    def_maybe_overloads(name, a1, a2, &a2);
    return *this;
  }

  /// \see boost::python::class_::def(char const* name, Fn fn, A1 const& a1, A2
  /// const& a2)
  template <class Fn, class A1, class A2>
  self& def(char const* name, Fn fn, A1 const& a1, A2 const& a2) {
    def_impl(bp::detail::unwrap_wrapper((W*)0), name, fn,
             bp::detail::def_helper<A1, A2>(a1, a2), &fn);

    return *this;
  }

  /// \see boost::python::class_::def(char const* name, Fn fn, A1 const& a1, A2
  /// const& a2, A3 const& a3)
  template <class Fn, class A1, class A2, class A3>
  self& def(char const* name, Fn fn, A1 const& a1, A2 const& a2, A3 const& a3) {
    def_impl(bp::detail::unwrap_wrapper((W*)0), name, fn,
             bp::detail::def_helper<A1, A2, A3>(a1, a2, a3), &fn);

    return *this;
  }

 private:
  /// \see boost::python::class_::def_impl(T*, char const* name, Fn fn, Helper
  /// const& helper, ...)
  template <class T, class Fn, class Helper>
  inline void def_impl(T*, char const* name, Fn fn, Helper const& helper, ...) {
    bp::objects::add_to_namespace(
        m_object, name,
        make_function(fn, helper.policies(), helper.keywords(),
                      bp::detail::get_signature(fn, (T*)0)),
        helper.doc());

    def_default(name, fn, helper,
                boost::mpl::bool_<Helper::has_default_implementation>());
  }

  /// \see boost::python::class_::def_default(char const* name, Fn, Helper
  /// const& helper, boost::mpl::bool_<true>)
  template <class Fn, class Helper>
  inline void def_default(char const* name, Fn, Helper const& helper,
                          boost::mpl::bool_<true>) {
    bp::detail::error::virtual_function_default<
        W, Fn>::must_be_derived_class_member(helper.default_implementation());

    bp::objects::add_to_namespace(
        m_object, name,
        make_function(helper.default_implementation(), helper.policies(),
                      helper.keywords()));
  }

  /// \see boost::python::class_::def_default(char const*, Fn, Helper const&,
  /// boost::mpl::bool_<false>)
  template <class Fn, class Helper>
  inline void def_default(char const*, Fn, Helper const&,
                          boost::mpl::bool_<false>) {}

  /// \see boost::python::class_::def_maybe_overloads(char const* name, SigT
  /// sig,OverloadsT const& overloads,bp::detail::overloads_base const*)
  template <class OverloadsT, class SigT>
  void def_maybe_overloads(char const* name, SigT sig,
                           OverloadsT const& overloads,
                           bp::detail::overloads_base const*)

  {
    bp::detail::define_with_defaults(name, overloads, *this,
                                     bp::detail::get_signature(sig));
  }

  /// \see boost::python::class_::def_maybe_overloads(char const* name, Fn fn,
  /// A1 const& a1, ...)
  template <class Fn, class A1>
  void def_maybe_overloads(char const* name, Fn fn, A1 const& a1, ...) {
    def_impl(bp::detail::unwrap_wrapper((W*)0), name, fn,
             bp::detail::def_helper<A1>(a1), &fn);
  }

 private:
  bp::object m_object;
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_registration_class_hpp__
