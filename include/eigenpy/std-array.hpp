///
/// Copyright (c) 2023-2024 CNRS INRIA
///

#ifndef __eigenpy_utils_std_array_hpp__
#define __eigenpy_utils_std_array_hpp__

#include <boost/python/suite/indexing/indexing_suite.hpp>
#include "eigenpy/std-vector.hpp"

#include <array>

namespace eigenpy {

template <typename Container, bool NoProxy, class SliceAllocator,
          class DerivedPolicies>
class array_indexing_suite;
namespace details {

template <typename Container, bool NoProxy, class SliceAllocator>
class final_array_derived_policies
    : public array_indexing_suite<
          Container, NoProxy, SliceAllocator,
          final_array_derived_policies<Container, NoProxy, SliceAllocator>> {};
}  // namespace details

template <typename Container, bool NoProxy = false,
          class SliceAllocator = std::allocator<typename Container::value_type>,
          class DerivedPolicies = details::final_array_derived_policies<
              Container, NoProxy, SliceAllocator>>
class array_indexing_suite
    : public bp::vector_indexing_suite<Container, NoProxy, DerivedPolicies> {
 public:
  typedef typename Container::value_type data_type;
  typedef typename Container::value_type key_type;
  typedef typename Container::size_type index_type;
  typedef typename Container::size_type size_type;
  typedef typename Container::difference_type difference_type;
  typedef std::vector<data_type, SliceAllocator> slice_vector_type;
  static constexpr std::size_t Size = std::tuple_size<Container>{};

  template <class Class>
  static void extension_def(Class &) {}

  // throws exception
  static void delete_item(Container &, index_type) {
    PyErr_SetString(PyExc_NotImplementedError,
                    "Cannot delete item from std::array type.");
    bp::throw_error_already_set();
  }

  // throws exception
  static void delete_slice(Container &, index_type, index_type) {
    PyErr_SetString(PyExc_NotImplementedError,
                    "Cannot delete slice from std::array type.");
    bp::throw_error_already_set();
  }

  static void set_slice(Container &container, index_type from, index_type to,
                        data_type const &v) {
    if (from >= to) {
      PyErr_SetString(PyExc_NotImplementedError,
                      "Setting this slice would insert into an std::array, "
                      "which is not supported.");
      bp::throw_error_already_set();
    } else {
      std::fill(container.begin() + from, container.begin() + to, v);
    }
  }

  template <class Iter>
  static void set_slice(Container &container, index_type from, index_type to,
                        Iter first, Iter last) {
    if (from >= to) {
      PyErr_SetString(PyExc_NotImplementedError,
                      "Setting this slice would insert into an std::array, "
                      "which is not supported.");
      bp::throw_error_already_set();
    } else {
      if (long(to - from) == std::distance(first, last)) {
        std::copy(first, last, container.begin() + from);
      } else {
        PyErr_SetString(PyExc_NotImplementedError,
                        "Size of std::array slice and size of right-hand side "
                        "iterator are incompatible.");
        bp::throw_error_already_set();
      }
    }
  }

  static bp::object get_slice(Container &container, index_type from,
                              index_type to) {
    if (from > to) return bp::object(slice_vector_type());
    slice_vector_type out;
    for (size_t i = from; i < to; i++) {
      out.push_back(container[i]);
    }
    return bp::object(std::move(out));
  }
};

/// \brief Expose an std::array (a C++11 fixed-size array) from a given type
/// \tparam array_type std::array type to expose
/// \tparam NoProxy When set to false, the elements will be copied when
/// returned to Python.
/// \tparam SliceAllocator Allocator type to use for slices of std::array type
/// accessed using e.g. __getitem__[0:4] in Python. These slices are returned as
/// std::vector (dynamic size).
template <typename array_type, bool NoProxy = false,
          class SliceAllocator =
              std::allocator<typename array_type::value_type>>
struct StdArrayPythonVisitor {
  typedef typename array_type::value_type value_type;

  static ::boost::python::list tolist(array_type &self, const bool deep_copy) {
    return details::build_list<array_type, NoProxy>::run(self, deep_copy);
  }

  static void expose(const std::string &class_name,
                     const std::string &doc_string = "") {
    expose(class_name, doc_string, EmptyPythonVisitor());
  }

  template <typename DerivedVisitor>
  static void expose(const std::string &class_name,
                     const bp::def_visitor<DerivedVisitor> &visitor) {
    expose(class_name, "", visitor);
  }

  template <typename DerivedVisitor>
  static void expose(const std::string &class_name,
                     const std::string &doc_string,
                     const bp::def_visitor<DerivedVisitor> &visitor) {
    if (!register_symbolic_link_to_registered_type<array_type>()) {
      bp::class_<array_type> cl(class_name.c_str(), doc_string.c_str());
      cl.def(bp::init<const array_type &>(bp::args("self", "other"),
                                          "Copy constructor"));
      cl.def(IdVisitor<array_type>());

      array_indexing_suite<array_type, NoProxy, SliceAllocator> indexing_suite;
      cl.def(indexing_suite)
          .def(visitor)
          .def("tolist", tolist,
               (bp::arg("self"), bp::arg("deep_copy") = false),
               "Returns the std::array as a Python list.");
    }
  }
};

/// Exposes std::array<MatrixType, Size>
template <typename MatrixType, std::size_t Size>
void exposeStdArrayEigenSpecificType(const char *name) {
  std::ostringstream oss;
  oss << "StdArr";
  oss << Size << "_" << name;
  typedef std::array<MatrixType, Size> array_type;
  StdArrayPythonVisitor<array_type, false,
                        Eigen::aligned_allocator<MatrixType>>::
      expose(oss.str(),
             details::overload_base_get_item_for_std_vector<array_type>());
}

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_std_array_hpp__
