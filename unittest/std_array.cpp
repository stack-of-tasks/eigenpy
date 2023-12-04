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

struct test_struct {
  std::array<int, 3> integs;
  std::array<VectorXd, 2> vecs;
  test_struct() {
    integs = {42, 3, -1};
    vecs[0].setRandom(4);  // 4 randoms between [-1,1]
    vecs[1].setZero(11);   // 11 zeroes
  }
};

BOOST_PYTHON_MODULE(std_array) {
  using namespace eigenpy;

  enableEigenPy();

  StdArrayPythonVisitor<std::array<int, 3>, true>::expose("StdArr3_int");
  StdVectorPythonVisitor<std::vector<int>, true>::expose("StdVec_int");

  exposeStdArrayEigenSpecificType<VectorXd, 2>("VectorXd");
  exposeStdArrayEigenSpecificType<VectorXd, 3>("VectorXd");
  exposeStdVectorEigenSpecificType<VectorXd>("VectorXd");

  bp::def("get_arr_3_ints", get_arr_3_ints);
  bp::def("get_arr_3_vecs", get_arr_3_vecs);

  bp::class_<test_struct>("test_struct", bp::init<>(bp::args("self")))
      .def_readwrite("integs", &test_struct::integs)
      .def_readwrite("vecs", &test_struct::vecs);
}
