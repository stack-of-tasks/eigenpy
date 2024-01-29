//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_utils_boost_variant_hpp__
#define __eigenpy_utils_boost_variant_hpp__

#include <boost/python.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/for_each.hpp>

namespace eigenpy {

namespace details {

/// Convert boost::variant<class...> alternative to a Python object.
/// This converter copy the alternative.
template <typename Variant>
struct BoostVariantValueToObject : boost::static_visitor<PyObject*> {
  typedef Variant variant_type;

  static result_type convert(const variant_type& gm) {
    return apply_visitor(BoostVariantValueToObject(), gm);
  }

  template <typename T>
  result_type operator()(T& t) const {
    return boost::python::incref(boost::python::object(t).ptr());
  }
};

/// Convert boost::variant<class...> alternative reference to a Python object.
/// This converter return the alternative reference.
/// The code that create the reference holder is taken from
/// \see boost::python::to_python_indirect.
template <typename Variant>
struct BoostVariantRefToObject : boost::static_visitor<PyObject*> {
  typedef Variant variant_type;

  static result_type convert(const variant_type& gm) {
    return apply_visitor(BoostVariantRefToObject(), gm);
  }

  template <typename T>
  result_type operator()(T& t) const {
    return boost::python::detail::make_reference_holder::execute(&t);
  }
};

/// Converter used in \see ReturnInternalBoostVariant.
/// This is inspired by \see boost::python::reference_existing_object.
/// It will call \see BoostVariantRefToObject to extract the alternative
/// reference.
template <typename Variant>
struct BoostVariantConverter {
  typedef Variant variant_type;

  template <class T>
  struct apply {
    struct type {
      PyObject* operator()(const variant_type& gm) const {
        return BoostVariantRefToObject<variant_type>::convert(gm);
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
struct BoostVariantImplicitlyConvertible {
  typedef Variant variant_type;

  template <class T>
  void operator()(T) {
    boost::python::implicitly_convertible<T, variant_type>();
  }
};

}  // namespace details

/// Variant of \see boost::python::return_internal_reference that
/// extract boost::variant<class...> alternative reference before
/// converting it into a PyObject
template <typename Variant>
struct ReturnInternalBoostVariant : boost::python::return_internal_reference<> {
  typedef Variant variant_type;

  typedef details::BoostVariantConverter<variant_type> result_converter;
};

/// Define a defaults converter to convert a boost::variant alternative to a
/// Python object by copy and to convert implicitly an alternative to a
/// boost::variant.
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
///     typedef eigenpy::BoostVariantConvertor<MyVariant> Convertor;
///     Convertor::registration();
///
///     boost::python::class_<VariantHolder>("VariantHolder", bp::init<>())
///       .add_property("variant",
///         bp::make_getter(&VariantHolder::variant,
///                         Convertor::return_internal_reference()),
///         bp::make_setter(&VariantHolder::variant));
///   }
template <typename Variant>
struct BoostVariantConvertor {
  typedef Variant variant_type;
  typedef ReturnInternalBoostVariant<variant_type> return_internal_reference;

  static void registration() {
    typedef details::BoostVariantValueToObject<variant_type> variant_to_value;
    boost::python::to_python_converter<variant_type, variant_to_value>();
    boost::mpl::for_each<typename variant_type::types>(
        details::BoostVariantImplicitlyConvertible<variant_type>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_boost_variant_hpp__
