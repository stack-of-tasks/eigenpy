/// Copyright (c) 2016-2022 CNRS INRIA
/// This file was taken from Pinocchio (header
/// <pinocchio/bindings/python/utils/std-vector.hpp>)
///

#ifndef __eigenpy_utils_map_hpp__
#define __eigenpy_utils_map_hpp__

#include "eigenpy/pickle-vector.hpp"

#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/to_python_converter.hpp>
#include <map>

namespace eigenpy {
namespace details {
template <typename Container>
struct overload_base_get_item_for_std_map
    : public boost::python::def_visitor<
          overload_base_get_item_for_std_map<Container> > {
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

}  // namespace details

///////////////////////////////////////////////////////////////////////////////
// The following snippet of code has been taken from the header
// https://github.com/loco-3d/crocoddyl/blob/v2.1.0/bindings/python/crocoddyl/utils/map-converter.hpp
// The Crocoddyl library is written by Carlos Mastalli, Nicolas Mansard and
// Rohan Budhiraja.
///////////////////////////////////////////////////////////////////////////////

namespace python {

namespace bp = boost::python;

/**
 * @brief Create a pickle interface for the std::map
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
      map[key] = val;
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

/**
 * @brief Expose an std::map from a type given as template argument.
 *
 * @param[in] T          Type to expose as std::map<T>.
 * @param[in] Compare    Type for the Compare in std::map<T,Compare,Allocator>.
 * @param[in] Allocator  Type for the Allocator in
 * std::map<T,Compare,Allocator>.
 * @param[in] NoProxy    When set to false, the elements will be copied when
 * returned to Python.
 */
template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T> >,
          bool NoProxy = false>
struct StdMapPythonVisitor
    : public bp::map_indexing_suite<
          typename std::map<Key, T, Compare, Allocator>, NoProxy>,
      public dict_to_map<std::map<Key, T, Compare, Allocator> > {
  typedef std::map<Key, T, Compare, Allocator> Container;
  typedef dict_to_map<Container> FromPythonDictConverter;

  static void expose(const std::string& class_name,
                     const std::string& doc_string = "") {
    namespace bp = bp;

    bp::class_<Container>(class_name.c_str(), doc_string.c_str())
        .def(StdMapPythonVisitor())
        .def("todict", &FromPythonDictConverter::todict, bp::arg("self"),
             "Returns the std::map as a Python dictionary.")
        .def_pickle(PickleMap<Container>());
    // Register conversion
    FromPythonDictConverter::register_converter();
  }
};

}  // namespace python
}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_map_hpp__
