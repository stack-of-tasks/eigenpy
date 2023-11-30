/// Copyright (c) 2023 CNRS INRIA

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
          final_array_derived_policies<Container, NoProxy, SliceAllocator> > {};
}  // namespace details

template <typename Container, bool NoProxy = false,
          class SliceAllocator = std::allocator<typename Container::value_type>,
          class DerivedPolicies = details::final_array_derived_policies<
              Container, NoProxy, SliceAllocator> >
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

  template <class Iter>
  static void extend(Container &, Iter, Iter) {}

  static void append(Container &, data_type const &) {}
  // no-op
  static void delete_item(Container &, index_type) {}
  // no-op
  // no-op
  static void delete_slice(Container &, index_type, index_type) {}

  static void set_slice(Container &container, index_type from, index_type to,
                        data_type const &v) {
    if (from > to) {
      return;
    } else {
      std::fill(container.begin() + from, container.begin() + to, v);
    }
  }

  template <class Iter>
  static void set_slice(Container &container, index_type from, index_type to,
                        Iter first, Iter last) {
    if (from > to) {
      return;
    } else {
      std::copy(first, last, container.begin() + from);
    }
  }

  static bp::object get_slice(Container &container, index_type from,
                              index_type to) {
    if (from > to) return bp::object(std::array<data_type, 0>());
    size_t size = to - from + 1;  // will be >= 0
    slice_vector_type out;
    for (size_t i = 0; i < size; i++) {
      out.push_back(container[i]);
    }
    return bp::object(std::move(out));
  }
};

template <typename array_type, bool NoProxy = false>
struct StdArrayPythonVisitor {
  typedef typename array_type::value_type value_type;
  /// Fixed size of the array, known at compile time
  static constexpr std::size_t Size = std::tuple_size<array_type>{};

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
                     const bp::def_visitor<DerivedVisitor> &) {
    if (!register_symbolic_link_to_registered_type<array_type>()) {
      bp::class_<array_type> cl(class_name.c_str(), doc_string.c_str());
      cl.def(bp::init<const array_type &>(bp::args("self", "other"),
                                          "Copy constructor"));

      array_indexing_suite<array_type, NoProxy> indexing_suite;
      cl.def(indexing_suite);
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
  StdArrayPythonVisitor<array_type>::expose(
      oss.str(), details::overload_base_get_item_for_std_vector<array_type>());
}

}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_std_array_hpp__
