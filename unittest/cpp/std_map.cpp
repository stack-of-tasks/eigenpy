/// @file
/// @copyright Copyright 2023 CNRS INRIA

#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-map.hpp>
#include <boost/unordered_map.hpp>

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

template <typename T1>
boost::unordered_map<std::string, T1> copy_boost(
    const boost::unordered_map<std::string, T1>& obj) {
  return obj;
}

struct X {
  X() = delete;
  X(int x) : val(x) {}
  int val;
};

BOOST_PYTHON_MODULE(std_map) {
  eigenpy::enableEigenPy();

  eigenpy::StdMapPythonVisitor<
      std::string, double, std::less<std::string>,
      std::allocator<std::pair<const std::string, double>>,
      true>::expose("StdMap_Double");

  eigenpy::GenericMapVisitor<boost::unordered_map<std::string, int>>::expose(
      "boost_map_int");

  using StdMap_X = std::map<std::string, X>;
  bp::class_<X>("X", bp::init<int>()).def_readwrite("val", &X::val);

  // this just needs to compile
  eigenpy::GenericMapVisitor<StdMap_X>::expose(
      "StdMap_X", eigenpy::overload_base_get_item_for_map<StdMap_X>());

  bp::def("std_map_to_dict", std_map_to_dict<double>);
  bp::def("copy", copy<double>);
  bp::def("copy_boost", copy_boost<int>);
  bp::def("copy_X", +[](const StdMap_X& m) { return m; });
}
