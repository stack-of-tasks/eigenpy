/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2022, INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"
// include main header first
#include "eigenpy/eigen-from-python.hpp"

using namespace Eigen;
using namespace eigenpy;

template <typename MatType>
void printMatrix(const Eigen::Ref<const MatType> mat) {
  if (MatType::IsVectorAtCompileTime) std::cout << "isVector" << std::endl;
  std::cout << "size: cols " << mat.cols() << " rows " << mat.rows()
            << std::endl;
  std::cout << mat << std::endl;
}

template <typename VecType>
void printVector(const Eigen::Ref<const VecType>& vec) {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(VecType);
  printMatrix(vec);
}

template <typename MatType>
void setOnes(Eigen::Ref<MatType> mat) {
  mat.setOnes();
}

template <typename MatType>
Eigen::Ref<MatType> getBlock(Eigen::Ref<MatType> mat, Eigen::DenseIndex i,
                             Eigen::DenseIndex j, Eigen::DenseIndex n,
                             Eigen::DenseIndex m) {
  return mat.block(i, j, n, m);
}

template <typename MatType>
Eigen::Ref<MatType> editBlock(Eigen::Ref<MatType> mat, Eigen::DenseIndex i,
                              Eigen::DenseIndex j, Eigen::DenseIndex n,
                              Eigen::DenseIndex m) {
  typename Eigen::Ref<MatType>::BlockXpr B = mat.block(i, j, n, m);
  int k = 0;
  for (int i = 0; i < B.rows(); ++i) {
    for (int j = 0; j < B.cols(); ++j) {
      B(i, j) = k++;
    }
  }
  std::cout << "B:\n" << B << std::endl;
  return mat;
}

template <typename MatType>
void fill(Eigen::Ref<MatType> mat, const typename MatType::Scalar& value) {
  mat.fill(value);
}

template <typename MatType>
Eigen::Ref<MatType> asRef(const int rows, const int cols) {
  static MatType mat(rows, cols);
  std::cout << "mat:\n" << mat << std::endl;
  return mat;
}

template <typename MatType>
Eigen::Ref<MatType> asRef(Eigen::Ref<MatType> mat) {
  return Eigen::Ref<MatType>(mat);
}

template <typename MatType>
const Eigen::Ref<const MatType> asConstRef(Eigen::Ref<MatType> mat) {
  return Eigen::Ref<const MatType>(mat);
}

struct modify_block {
  MatrixXd J;
  modify_block() : J(10, 10) { J.setZero(); }
  void modify(int n, int m) { call(J.topLeftCorner(n, m)); }
  virtual void call(Eigen::Ref<MatrixXd> mat) = 0;
};

struct modify_wrap : modify_block, bp::wrapper<modify_block> {
  modify_wrap() : modify_block() {}
  void call(Eigen::Ref<MatrixXd> mat) { this->get_override("call")(mat); }
};

struct has_ref_member {
  MatrixXd J;
  Eigen::Ref<MatrixXd> Jref;
  has_ref_member() : J(4, 4), Jref(J.topRightCorner(3, 3)) { J.setZero(); }
};

BOOST_PYTHON_MODULE(eigen_ref) {
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  bp::def("printMatrix", printMatrix<Vector3d>);
  bp::def("printMatrix", printMatrix<VectorXd>);
  bp::def("printMatrix", printMatrix<MatrixXd>);

  bp::def("printVector", printVector<VectorXd>);
  bp::def("printRowVector", printVector<RowVectorXd>);

  bp::def("setOnes", setOnes<Vector3d>);
  bp::def("setOnes", setOnes<VectorXd>);
  bp::def("setOnes", setOnes<MatrixXd>);

  bp::def("fillVec3", fill<Vector3d>);
  bp::def("fillVec", fill<VectorXd>);
  bp::def("fill", fill<MatrixXd>);

  bp::def("asRef",
          (Eigen::Ref<MatrixXd>(*)(const int, const int))asRef<MatrixXd>);
  bp::def("asRef",
          (Eigen::Ref<MatrixXd>(*)(Eigen::Ref<MatrixXd>))asRef<MatrixXd>);
  bp::def("asConstRef", (const Eigen::Ref<const MatrixXd> (*)(
                            Eigen::Ref<MatrixXd>))asConstRef<MatrixXd>);

  bp::def("getBlock", &getBlock<MatrixXd>);
  bp::def("editBlock", &editBlock<MatrixXd>);

  bp::class_<modify_wrap, boost::noncopyable>("modify_block", bp::init<>())
      .def_readonly("J", &modify_block::J)
      .def("modify", &modify_block::modify)
      .def("call", bp::pure_virtual(&modify_wrap::call));

  bp::class_<has_ref_member, boost::noncopyable>("has_ref_member", bp::init<>())
      .def_readonly("J", &has_ref_member::J)
      .add_property(
          "Jref",
          bp::make_getter(&has_ref_member::Jref,
                          bp::return_value_policy<bp::return_by_value>()));
  // can't return Eigen::Ref by reference but by value
  // (def_readonly creates a by-reference getter)
}
