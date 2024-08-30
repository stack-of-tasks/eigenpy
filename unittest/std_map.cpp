/// @file
/// @copyright Copyright 2023 CNRS INRIA

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-map.hpp>
#include <iostream>

namespace bp = boost::python;

template <typename T1>
bp::dict std_map_to_dict(const std::map<std::string, T1>& map) {
  bp::dict dictionnary;
  for (auto const& x : map) {
    dictionnary[x.first] = x.second;
  }
  return dictionnary;
}

template <typename T1>
std::map<std::string, T1> copy(const std::map<std::string, T1>& map) {
  std::map<std::string, T1> out = map;
  return out;
}

BOOST_PYTHON_MODULE(std_map) {
  eigenpy::enableEigenPy();

  eigenpy::python::StdMapPythonVisitor<
      std::string, double, std::less<std::string>,
      std::allocator<std::pair<const std::string, double> >,
      true>::expose("StdMap_Double");

  bp::def("std_map_to_dict", std_map_to_dict<double>);
  bp::def("copy", copy<double>);
}
