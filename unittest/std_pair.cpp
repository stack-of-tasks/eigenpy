/// @file
/// @copyright Copyright 2023 CNRS INRIA

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-pair.hpp>

namespace bp = boost::python;

template <typename T1, typename T2>
bp::tuple std_pair_to_tuple(const std::pair<T1, T2>& pair) {
  return bp::make_tuple(pair.first, pair.second);
}

template <typename T1, typename T2>
std::pair<T1, T2> copy(const std::pair<T1, T2>& pair) {
  return pair;
}

template <typename T1, typename T2>
const std::pair<T1, T2>& passthrough(const std::pair<T1, T2>& pair) {
  return pair;
}

BOOST_PYTHON_MODULE(std_pair) {
  eigenpy::enableEigenPy();

  typedef std::pair<int, double> PairType;
  eigenpy::StdPairConverter<PairType>::registration();

  bp::def("std_pair_to_tuple", std_pair_to_tuple<int, double>);
  bp::def("copy", copy<int, double>);
  bp::def("passthrough", passthrough<int, double>,
          bp::return_value_policy<bp::copy_const_reference>());
}
