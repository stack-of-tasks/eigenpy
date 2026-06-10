/// Copyright (c) 2016-2024 CNRS INRIA
/// This file was originally taken from Pinocchio (header
/// <pinocchio/bindings/python/utils/std-vector.hpp>)
///

#ifndef __eigenpy_map_hpp__
#define __eigenpy_map_hpp__

#include "eigenpy/pickle-vector.hpp"
#include "eigenpy/registration.hpp"
#include "eigenpy/utils/empty-visitor.hpp"

#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/to_python_converter.hpp>

namespace eigenpy {

/// \brief Change the behavior of indexing (method __getitem__ in Python).
/// This is suitable e.g. for container of Eigen matrix objects if you want to
/// mutate them.
/// \sa overload_base_get_item_for_std_vector
template <typename Container>
struct overload_base_get_item_for_map
    : public boost::python::def_visitor<
          overload_base_get_item_for_map<Container>> {
  typedef typename Container::value_type value_type;
  typedef typename Container::value_type::second_type data_type;
  typedef typename Container::key_type key_type;
  typedef typename Container::key_type index_type;

  template <class Class>
  void visit(Class& cl) const {
    cl.def("__getitem__", &base_get_item);
  }

 private:
  static boost::python::object base_get_item(
      boost::python::back_reference<Container&> container, PyObject* i_) {
    index_type idx = convert_index(container.get(), i_);
    typename Container::iterator i = container.get().find(idx);
    if (i == container.get().end()) {
      PyErr_SetString(PyExc_KeyError, "Invalid key");
      boost::python::throw_error_already_set();
    }

    typename boost::python::to_python_indirect<
        data_type&, boost::python::detail::make_reference_holder>
        convert;
    return boost::python::object(boost::python::handle<>(convert(i->second)));
  }

  static index_type convert_index(Container& /*container*/, PyObject* i_) {
    boost::python::extract<key_type const&> i(i_);
    if (i.check()) {
      return i();
    } else {
      boost::python::extract<key_type> i(i_);
      if (i.check()) return i();
    }

    PyErr_SetString(PyExc_TypeError, "Invalid index type");
    boost::python::throw_error_already_set();
    return index_type();
  }
};

///////////////////////////////////////////////////////////////////////////////
// The following snippet of code has been taken from the header
// https://github.com/loco-3d/crocoddyl/blob/v2.1.0/bindings/python/crocoddyl/utils/map-converter.hpp
// The Crocoddyl library is written by Carlos Mastalli, Nicolas Mansard and
// Rohan Budhiraja.
///////////////////////////////////////////////////////////////////////////////

namespace bp = boost::python;

/**
 * @brief Create a pickle interface for the map type
 *
 * @param[in] Container  Map type to be pickled
 * \sa Pickle
 */
template <typename Container>
struct PickleMap : public PickleVector<Container> {
  static void setstate(bp::object op, bp::tuple tup) {
    Container& o = bp::extract<Container&>(op)();
    bp::stl_input_iterator<typename Container::value_type> begin(tup[0]), end;
    o.insert(begin, end);
  }
};

/// Conversion from dict to map solution proposed in
/// https://stackoverflow.com/questions/6116345/boostpython-possible-to-automatically-convert-from-dict-stdmap
/// This template encapsulates the conversion machinery.
template <typename Container>
struct dict_to_map {
  static void register_converter() {
    bp::converter::registry::push_back(&dict_to_map::convertible,
                                       &dict_to_map::construct,
                                       bp::type_id<Container>());
  }

  /// Check if conversion is possible
  static void* convertible(PyObject* object) {
    // Check if it is a list
    if (!PyObject_GetIter(object)) return 0;
    return object;
  }

  /// Perform the conversion
  static void construct(PyObject* object,
                        bp::converter::rvalue_from_python_stage1_data* data) {
    // convert the PyObject pointed to by `object` to a bp::dict
    bp::handle<> handle(bp::borrowed(object));  // "smart ptr"
    bp::dict dict(handle);

    // get a pointer to memory into which we construct the map
    // this is provided by the Python runtime
    typedef bp::converter::rvalue_from_python_storage<Container> storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    // placement-new allocate the result
    new (storage) Container();

    // iterate over the dictionary `dict`, fill up the map `map`
    Container& map(*(static_cast<Container*>(storage)));
    bp::list keys(dict.keys());
    int keycount(static_cast<int>(bp::len(keys)));
    for (int i = 0; i < keycount; ++i) {
      // get the key
      bp::object keyobj(keys[i]);
      bp::extract<typename Container::key_type> keyproxy(keyobj);
      if (!keyproxy.check()) {
        PyErr_SetString(PyExc_KeyError, "Bad key type");
        bp::throw_error_already_set();
      }
      typename Container::key_type key = keyproxy();

      // get the corresponding value
      bp::object valobj(dict[keyobj]);
      bp::extract<typename Container::mapped_type> valproxy(valobj);
      if (!valproxy.check()) {
        PyErr_SetString(PyExc_ValueError, "Bad value type");
        bp::throw_error_already_set();
      }
      typename Container::mapped_type val = valproxy();
      map.emplace(key, val);
    }

    // remember the location for later
    data->convertible = storage;
  }

  static bp::dict todict(Container& self) {
    bp::dict dict;
    typename Container::const_iterator it;
    for (it = self.begin(); it != self.end(); ++it) {
      dict.setdefault(it->first, it->second);
    }
    return dict;
  }
};

/// Policies which handle the non-default constructible case
/// and set_item() using emplace().
template <class Container, bool NoProxy>
struct emplace_set_derived_policies
    : bp::map_indexing_suite<Container, NoProxy,
                             emplace_set_derived_policies<Container, NoProxy>> {
  typedef typename Container::key_type index_type;
  typedef typename Container::value_type::second_type data_type;
  typedef typename Container::value_type value_type;
  using DerivedPolicies =
      bp::detail::final_map_derived_policies<Container, NoProxy>;

  template <class Class>
  static void extension_def(Class& cl) {
    //  Wrap the map's element (value_type)
    std::string elem_name = "map_indexing_suite_";
    bp::object class_name(cl.attr("__name__"));
    bp::extract<std::string> class_name_extractor(class_name);
    elem_name += class_name_extractor();
    elem_name += "_entry";
    namespace mpl = boost::mpl;

    typedef typename mpl::if_<
        mpl::and_<boost::is_class<data_type>, mpl::bool_<!NoProxy>>,
        bp::return_internal_reference<>, bp::default_call_policies>::type
        get_data_return_policy;

    bp::class_<value_type>(elem_name.c_str(), bp::no_init)
        .def("__repr__", &DerivedPolicies::print_elem)
        .def("data", &DerivedPolicies::get_data, get_data_return_policy())
        .def("key", &DerivedPolicies::get_key);
  }

  static void set_item(Container& container, index_type i, data_type const& v) {
    container.emplace(i, v);
  }
};

/**
 * @brief Expose the map-like container, e.g. (std::map).
 *
 * @param[in] Container  Container to expose.
 * @param[in] NoProxy    When set to false, the elements will be copied when
 * returned to Python.
 */
template <class Container, bool NoProxy = false>
struct GenericMapVisitor
    : public emplace_set_derived_policies<Container, NoProxy>,
      public dict_to_map<Container> {
  typedef dict_to_map<Container> FromPythonDictConverter;

  template <typename DerivedVisitor>
  static void expose(const std::string& class_name,
                     const std::string& doc_string,
                     const bp::def_visitor<DerivedVisitor>& visitor) {
    namespace bp = bp;

    if (!register_symbolic_link_to_registered_type<Container>()) {
      bp::class_<Container>(class_name.c_str(), doc_string.c_str())
          .def(GenericMapVisitor())
          .def("todict", &FromPythonDictConverter::todict, bp::arg("self"),
               "Returns the map type as a Python dictionary.")
          .def_pickle(PickleMap<Container>())
          .def(visitor);
      // Register conversion
      FromPythonDictConverter::register_converter();
    }
  }

  static void expose(const std::string& class_name,
                     const std::string& doc_string = "") {
    expose(class_name, doc_string, EmptyPythonVisitor());
  }

  template <typename DerivedVisitor>
  static void expose(const std::string& class_name,
                     const bp::def_visitor<DerivedVisitor>& visitor) {
    expose(class_name, "", visitor);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_map_hpp__
