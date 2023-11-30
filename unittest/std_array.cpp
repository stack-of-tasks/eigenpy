/// @file
/// @copyright Copyright 2023 CNRS INRIA

#include "eigenpy/std-array.hpp"

using Eigen::VectorXd;

std::array<int, 3> get_arr_3_ints() { return {1, 2, 3}; }

std::array<VectorXd, 3> get_arr_3_vecs() {
  std::array<VectorXd, 3> out;
  out[0].setOnes(4);
  out[1].setZero(2);
  out[2].setRandom(10);
  return out;
}

BOOST_PYTHON_MODULE(std_array) {
  using namespace eigenpy;

  enableEigenPy();

  StdArrayPythonVisitor<std::array<int, 3> >::expose("StdArr3_int");
  exposeStdArrayEigenSpecificType<VectorXd, 3>("VectorXd");

  bp::def("get_arr_3_ints", get_arr_3_ints);
  bp::def("get_arr_3_vecs", get_arr_3_vecs);
}
