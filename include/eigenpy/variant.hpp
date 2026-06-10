//
// Copyright (c) 2024 INRIA
//

#ifndef __eigenpy_utils_variant_hpp__
#define __eigenpy_utils_variant_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/utils/traits.hpp"
#include "eigenpy/utils/python-compat.hpp"

#include <boost/python.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/vector.hpp>

#include <type_traits>

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
struct VariantAlternatives {};

template <typename Variant>
struct empty_variant {};

template <typename T>
struct is_empty_variant : std::false_type {};

#ifdef EIGENPY_WITH_CXX17_SUPPORT

/// std::variant implementation
template <typename ResultType, typename... Alternatives>
struct VariantVisitorType<ResultType, std::variant<Alternatives...>> {
  typedef std::variant<Alternatives...> variant_type;
  typedef ResultType result_type;

  template <typename Visitor, typename Visitable>
  static result_type visit(Visitor&& visitor, Visitable&& v) {
    return std::visit(std::forward<Visitor>(visitor),
                      std::forward<Visitable>(v));
  }

  result_type operator()(std::monostate) const {
    return bp::incref(bp::object().ptr());  // None
  }
};

template <typename... Alternatives>
struct VariantAlternatives<std::variant<Alternatives...>> {
  typedef boost::mpl::vector<Alternatives...> types;
};

template <typename... Alternatives>
struct empty_variant<std::variant<Alternatives...>> {
  typedef std::monostate type;
};

template <>
struct is_empty_variant<std::monostate> : std::true_type {};

#endif

/// boost::variant implementation
template <typename ResultType, typename... Alternatives>
struct VariantVisitorType<ResultType, boost::variant<Alternatives...>>
    : boost::static_visitor<ResultType> {
  typedef boost::variant<Alternatives...> variant_type;
  typedef ResultType result_type;

  template <typename Visitor, typename Visitable>
  static result_type visit(Visitor&& visitor, Visitable&& visitable) {
    return std::forward<Visitable>(visitable).apply_visitor(visitor);
  }

  result_type operator()(boost::blank) const {
    return bp::incref(bp::object().ptr());  // None
  }
};

template <typename... Alternatives>
struct VariantAlternatives<boost::variant<Alternatives...>> {
  typedef typename boost::variant<Alternatives...>::types types;
};

template <typename... Alternatives>
struct empty_variant<boost::variant<Alternatives...>> {
  typedef boost::blank type;
};

template <>
struct is_empty_variant<boost::blank> : std::true_type {};

/// Convert None to a {boost,std}::variant with boost::blank or std::monostate
/// value
template <typename Variant>
struct EmptyConvertible {
  static void registration() {
    bp::converter::registry::push_back(convertible, construct,
                                       bp::type_id<Variant>());
  }

  // convertible only for None
  static void* convertible(PyObject* obj) {
    return (obj == Py_None) ? obj : nullptr;
  };

  // construct in place
  static void construct(PyObject*,
                        bp::converter::rvalue_from_python_stage1_data* data) {
    void* storage =
        reinterpret_cast<bp::converter::rvalue_from_python_storage<Variant>*>(
            data)
            ->storage.bytes;
    new (storage) Variant(typename empty_variant<Variant>::type());
    data->convertible = storage;
  };
};

/// Implement convertible and expected_pytype for bool, integer and float
template <typename T, class Enable = void>
struct NumericConvertibleImpl {};

template <typename T>
struct NumericConvertibleImpl<
    T, typename std::enable_if<std::is_same<T, bool>::value>::type> {
  static void* convertible(PyObject* obj) {
    return PyBool_Check(obj) ? obj : nullptr;
  }

  static PyTypeObject const* expected_pytype() { return &PyBool_Type; }
};

template <typename T>
struct NumericConvertibleImpl<
    T, typename std::enable_if<!std::is_same<T, bool>::value &&
                               std::is_integral<T>::value>::type> {
  static void* convertible(PyObject* obj) {
    // PyLong return true for bool type
    return (PyInt_Check(obj) && !PyBool_Check(obj)) ? obj : nullptr;
  }

  static PyTypeObject const* expected_pytype() { return &PyLong_Type; }
};

template <typename T>
struct NumericConvertibleImpl<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  static void* convertible(PyObject* obj) {
    return PyFloat_Check(obj) ? obj : nullptr;
  }

  static PyTypeObject const* expected_pytype() { return &PyFloat_Type; }
};

/// Convert numeric type to Variant without ambiguity
template <typename T, typename Variant>
struct NumericConvertible {
  static void registration() {
    bp::converter::registry::push_back(
        &convertible, &bp::converter::implicit<T, Variant>::construct,
        bp::type_id<Variant>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
            ,
        &expected_pytype
#endif
    );
  }

  static void* convertible(PyObject* obj) {
    return NumericConvertibleImpl<T>::convertible(obj);
  }
  static PyTypeObject const* expected_pytype() {
    return NumericConvertibleImpl<T>::expected_pytype();
  }
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
    return bp::incref(bp::object(t).ptr());
  }

  using Base::operator();
};

/// Convert {boost,std}::variant<class...> alternative reference to a Python
/// object. This converter return the alternative reference. The code that
/// create the reference holder is taken from \see
/// bp::to_python_indirect.
template <typename Variant>
struct VariantRefToObject : VariantVisitorType<PyObject*, Variant> {
  typedef VariantVisitorType<PyObject*, Variant> Base;
  typedef typename Base::result_type result_type;
  typedef typename Base::variant_type variant_type;

  static result_type convert(const variant_type& v) {
    return Base::visit(VariantRefToObject(), v);
  }

  template <typename T,
            typename std::enable_if<is_python_primitive_type<T>::value,
                                    bool>::type = true>
  result_type operator()(T t) const {
    return bp::incref(bp::object(t).ptr());
  }

  template <typename T,
            typename std::enable_if<!is_python_primitive_type<T>::value,
                                    bool>::type = true>
  result_type operator()(T& t) const {
    return bp::detail::make_reference_holder::execute(&t);
  }

  /// Copy the object when it's None
  using Base::operator();
};

/// Converter used in \see ReturnInternalVariant.
/// This is inspired by \see bp::reference_existing_object.
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
        return bp::converter::registered_pytype<variant_type>::get_pytype();
      }
#endif
    };
  };
};

/// Convert an Alternative type to a Variant
template <typename Variant>
struct VariantConvertible {
  typedef Variant variant_type;

  template <class T, typename std::enable_if<is_empty_variant<T>::value,
                                             bool>::type = true>
  void operator()(T) {
    EmptyConvertible<variant_type>::registration();
  }

  template <class T, typename std::enable_if<!is_empty_variant<T>::value &&
                                                 std::is_arithmetic<T>::value,
                                             bool>::type = true>
  void operator()(T) {
    NumericConvertible<T, variant_type>::registration();
  }

  template <class T, typename std::enable_if<!is_empty_variant<T>::value &&
                                                 !std::is_arithmetic<T>::value,
                                             bool>::type = true>
  void operator()(T) {
    bp::implicitly_convertible<T, variant_type>();
  }
};

}  // namespace details

/// Variant of \see bp::return_internal_reference that
/// extract {boost,std}::variant<class...> alternative reference before
/// converting it into a PyObject
template <typename Variant>
struct ReturnInternalVariant : bp::return_internal_reference<> {
  typedef Variant variant_type;

  typedef details::VariantConverter<variant_type> result_converter;

  template <class ArgumentPackage>
  static PyObject* postcall(ArgumentPackage const& args_, PyObject* result) {
    // Don't run return_internal_reference postcall on primitive type
    if (PyInt_Check(result) || PyBool_Check(result) || PyFloat_Check(result) ||
        PyStr_Check(result) || PyComplex_Check(result)) {
      return result;
    }
    return bp::return_internal_reference<>::postcall(args_, result);
  }
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
///     bp::class_<Struct1>("Struct1", bp::init<>());
///     bp::class_<Struct2>("Struct1", bp::init<>())
///     typedef eigenpy::VariantConverter<MyVariant> Converter;
///     Converter::registration();
///
///     bp::class_<VariantHolder>("VariantHolder", bp::init<>())
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

    bp::to_python_converter<variant_type, variant_to_value>();
    boost::mpl::for_each<types>(details::VariantConvertible<variant_type>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_variant_hpp__
