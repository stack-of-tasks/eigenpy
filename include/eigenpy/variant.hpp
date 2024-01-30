//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_utils_variant_hpp__
#define __eigenpy_utils_variant_hpp__

#include "eigenpy/fwd.hpp"

#include <boost/python.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector.hpp>

#ifdef EIGENPY_WITH_CXX17_SUPPORT
#include <variant>
#endif

namespace eigenpy {

namespace details {

/// Allow to use std::variant and boost::variant with the same API
template <typename ResultType, typename Variant>
struct VariantVisitorType {};

/// Allow to get all alternatives in a boost::mpl vector
template <typename Variant>
struct VariantAlternatives{};

#ifdef EIGENPY_WITH_CXX17_SUPPORT

/// std::variant implementation
template <typename ResultType, typename... Alternatives>
struct VariantVisitorType<ResultType, std::variant<Alternatives...> > {
  typedef std::variant<Alternatives...> variant_type;
  typedef ResultType result_type;

  template <typename Visitor, typename Visitable>
  static result_type visit(Visitor&& visitor, Visitable&& v) {
    return std::visit(std::forward<Visitor>(visitor),
                      std::forward<Visitable>(v));
  }
};

template<typename... Alternatives>
struct VariantAlternatives<std::variant<Alternatives...>>{
  typedef boost::mpl::vector<Alternatives...> types;
};

#endif

/// boost::variant implementation
template <typename ResultType, typename... Alternatives>
struct VariantVisitorType<ResultType, boost::variant<Alternatives...> >
    : boost::static_visitor<ResultType> {
  typedef boost::variant<Alternatives...> variant_type;
  typedef ResultType result_type;

  template <typename Visitor, typename Visitable>
  static result_type visit(Visitor&& visitor, Visitable&& visitable) {
    return std::forward<Visitable>(visitable).apply_visitor(visitor);
  }
};

template<typename... Alternatives>
struct VariantAlternatives<boost::variant<Alternatives...>>{
  typedef typename boost::variant<Alternatives...>::types types;
};

/// Convert {boost,std}::variant<class...> alternative to a Python object.
/// This converter copy the alternative.
template <typename Variant>
struct VariantValueToObject : VariantVisitorType<PyObject*, Variant> {
  typedef VariantVisitorType<PyObject*, Variant> Base;
  typedef typename Base::result_type result_type;
  typedef typename Base::variant_type variant_type;

  static result_type convert(const variant_type& v) {
    return Base::visit(VariantValueToObject(), v);
  }

  template <typename T>
  result_type operator()(T& t) const {
    return boost::python::incref(boost::python::object(t).ptr());
  }
};

/// Convert {boost,std}::variant<class...> alternative reference to a Python
/// object. This converter return the alternative reference. The code that
/// create the reference holder is taken from \see
/// boost::python::to_python_indirect.
template <typename Variant>
struct VariantRefToObject : VariantVisitorType<PyObject*, Variant> {
  typedef VariantVisitorType<PyObject*, Variant> Base;
  typedef typename Base::result_type result_type;
  typedef typename Base::variant_type variant_type;

  static result_type convert(const variant_type& v) {
    return Base::visit(VariantRefToObject(), v);
  }

  template <typename T>
  result_type operator()(T& t) const {
    return boost::python::detail::make_reference_holder::execute(&t);
  }
};

/// Converter used in \see ReturnInternalVariant.
/// This is inspired by \see boost::python::reference_existing_object.
/// It will call \see VariantRefToObject to extract the alternative
/// reference.
template <typename Variant>
struct VariantConverter {
  typedef Variant variant_type;

  template <class T>
  struct apply {
    struct type {
      PyObject* operator()(const variant_type& v) const {
        return VariantRefToObject<variant_type>::convert(v);
      }

#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
      PyTypeObject const* get_pytype() const {
        return boost::python::converter::registered_pytype<
            variant_type>::get_pytype();
      }
#endif
    };
  };
};

/// Declare a variant alternative implicitly convertible to the variant
template <typename Variant>
struct VariantImplicitlyConvertible {
  typedef Variant variant_type;

  template <class T>
  void operator()(T) {
    boost::python::implicitly_convertible<T, variant_type>();
  }
};

}  // namespace details

/// Variant of \see boost::python::return_internal_reference that
/// extract {boost,std}::variant<class...> alternative reference before
/// converting it into a PyObject
template <typename Variant>
struct ReturnInternalVariant : boost::python::return_internal_reference<> {
  typedef Variant variant_type;

  typedef details::VariantConverter<variant_type> result_converter;
};

/// Define a defaults converter to convert a {boost,std}::variant alternative to
/// a Python object by copy and to convert implicitly an alternative to a
/// {boost,std}::variant.
///
/// Example:
///
///   typedef boost::variant<Struct1, Struct2> MyVariant;
///   struct VariantHolder {
///     MyVariant variant;
///   };
///   ...
///   void expose() {
///     boost::python::class_<Struct1>("Struct1", bp::init<>());
///     boost::python::class_<Struct2>("Struct1", bp::init<>())
///     typedef eigenpy::VariantConverter<MyVariant> Converter;
///     Converter::registration();
///
///     boost::python::class_<VariantHolder>("VariantHolder", bp::init<>())
///       .add_property("variant",
///         bp::make_getter(&VariantHolder::variant,
///                         Converter::return_internal_reference()),
///         bp::make_setter(&VariantHolder::variant));
///   }
template <typename Variant>
struct VariantConverter {
  typedef Variant variant_type;
  typedef ReturnInternalVariant<variant_type> return_internal_reference;

  static void registration() {
    typedef details::VariantValueToObject<variant_type> variant_to_value;
    typedef typename details::VariantAlternatives<variant_type>::types types;
    boost::python::to_python_converter<variant_type, variant_to_value>();
    boost::mpl::for_each<types>(
        details::VariantImplicitlyConvertible<variant_type>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_variant_hpp__
