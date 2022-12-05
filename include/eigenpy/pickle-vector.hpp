//
// Copyright (c) 2019-2020 CNRS INRIA
//

#ifndef __eigenpy_utils_pickle_vector_hpp__
#define __eigenpy_utils_pickle_vector_hpp__

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/tuple.hpp>

namespace eigenpy {
///
/// \brief Create a pickle interface for the std::vector
///
/// \tparam VecType Vector Type to pickle
///
template <typename VecType>
struct PickleVector : boost::python::pickle_suite {
  static boost::python::tuple getinitargs(const VecType&) {
    return boost::python::make_tuple();
  }

  static boost::python::tuple getstate(boost::python::object op) {
    return boost::python::make_tuple(
        boost::python::list(boost::python::extract<const VecType&>(op)()));
  }

  static void setstate(boost::python::object op, boost::python::tuple tup) {
    if (boost::python::len(tup) > 0) {
      VecType& o = boost::python::extract<VecType&>(op)();
      boost::python::stl_input_iterator<typename VecType::value_type> begin(
          tup[0]),
          end;
      while (begin != end) {
        o.push_back(*begin);
        ++begin;
      }
    }
  }

  static bool getstate_manages_dict() { return true; }
};
}  // namespace eigenpy

#endif  // ifndef __eigenpy_utils_pickle_vector_hpp__
