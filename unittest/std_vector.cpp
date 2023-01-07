/// @file
/// @copyright Copyright 2022, CNRS
/// @copyright Copyright 2022, INRIA
#include <ostream>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen-from-python.hpp"
#include "eigenpy/std-vector.hpp"

template <typename MatType,
          typename Allocator = Eigen::aligned_allocator<MatType> >
void printVectorOfMatrix(const std::vector<MatType, Allocator> &Ms) {
  const std::size_t n = Ms.size();
  for (std::size_t i = 0; i < n; i++) {
    std::cout << "el[" << i << "] =\n" << Ms[i] << '\n';
  }
}

template <typename MatType,
          typename Allocator = Eigen::aligned_allocator<MatType> >
std::vector<MatType, Allocator> copy(
    const std::vector<MatType, Allocator> &Ms) {
  std::vector<MatType, Allocator> out = Ms;
  return out;
}

template <typename MatType,
          typename Allocator = Eigen::aligned_allocator<MatType> >
void setZero(std::vector<MatType, Allocator> &Ms) {
  for (std::size_t i = 0; i < Ms.size(); i++) {
    Ms[i].setZero();
  }
}

BOOST_PYTHON_MODULE(std_vector) {
  namespace bp = boost::python;
  using namespace eigenpy;

  enableEigenPy();

  bp::def("printVectorOfMatrix", printVectorOfMatrix<Eigen::VectorXd>);
  bp::def("printVectorOfMatrix", printVectorOfMatrix<Eigen::MatrixXd>);

  bp::def("copyStdVector", copy<Eigen::MatrixXd>);
  bp::def("copyStdVector", copy<Eigen::VectorXd>);

  exposeStdVectorEigenSpecificType<Eigen::Matrix3d>("Mat3d");
  bp::def("printVectorOf3x3", printVectorOfMatrix<Eigen::Matrix3d>);
  bp::def("copyStdVec_3x3", copy<Eigen::Matrix3d>, bp::args("mats"));

  typedef Eigen::Ref<Eigen::MatrixXd> RefXd;
  StdVectorPythonVisitor<std::vector<RefXd>, true>::expose("StdVec_MatRef");
  bp::def("setZero", setZero<Eigen::MatrixXd>, "Sets the coeffs to 0.");
}
