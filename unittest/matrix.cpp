/*
 * Copyright 2014-2022 CNRS INRIA
 */

#include <iostream>

#include "eigenpy/eigenpy.hpp"

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vector1x1(const Scalar& value) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ReturnType;
  return ReturnType::Constant(1, value);
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix1x1(
    const Scalar& value) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ReturnType;
  return ReturnType::Constant(1, 1, value);
}

template <typename Scalar>
void matrix1x1_input(const Eigen::Matrix<Scalar, 1, 1>& mat) {
  std::cout << mat << std::endl;
}

Eigen::VectorXd emptyVector() {
  Eigen::VectorXd vec;
  vec.resize(0);
  return vec;
}

Eigen::MatrixXd emptyMatrix() { return Eigen::MatrixXd(0, 0); }

Eigen::MatrixXd naturals(int R, int C, bool verbose) {
  Eigen::MatrixXd mat(R, C);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) mat(r, c) = r * C + c;

  if (verbose) std::cout << "EigenMat = " << mat << std::endl;
  return mat;
}

Eigen::VectorXd naturals(int R, bool verbose) {
  Eigen::VectorXd mat(R);
  for (int r = 0; r < R; ++r) mat[r] = r;

  if (verbose) std::cout << "EigenMat = " << mat << std::endl;
  return mat;
}

Eigen::Matrix3d naturals(bool verbose) {
  Eigen::Matrix3d mat;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) mat(r, c) = r * 3 + c;

  if (verbose) std::cout << "EigenMat = " << mat << std::endl;
  return mat;
}

template <typename MatType>
Eigen::MatrixXd reflex(const MatType& M, bool verbose) {
  if (verbose) std::cout << "EigenMat = " << M << std::endl;
  return Eigen::MatrixXd(M);
}

template <typename MatrixDerived>
MatrixDerived base(const Eigen::MatrixBase<MatrixDerived>& m) {
  return m.derived();
}

template <typename MatrixDerived>
MatrixDerived plain(const Eigen::PlainObjectBase<MatrixDerived>& m) {
  return m.derived();
}

template <typename Scalar>
Eigen::Matrix<Scalar, 6, 6> matrix6(const Scalar& value) {
  typedef Eigen::Matrix<Scalar, 6, 6> ReturnType;
  return ReturnType::Constant(value);
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
generateRowMajorMatrix(const Eigen::DenseIndex rows,
                       const Eigen::DenseIndex cols) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrix;
  RowMajorMatrix A(rows, cols);
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Eigen::Map<Vector>(A.data(), A.size()) =
      Vector::LinSpaced(A.size(), 1, static_cast<Scalar>(A.size()));
  std::cout << "Matrix values:\n" << A << std::endl;
  return A;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>
generateRowMajorVector(const Eigen::DenseIndex size) {
  typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorVector;
  RowMajorVector A(size);
  A.setLinSpaced(size, 1, static_cast<Scalar>(size));
  std::cout << "Vector values: " << A.transpose() << std::endl;
  return A;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> generateColMajorMatrix(
    const Eigen::DenseIndex rows, const Eigen::DenseIndex cols) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> ColMajorMatrix;
  ColMajorMatrix A(rows, cols);
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Eigen::Map<Vector>(A.data(), A.size()) =
      Vector::LinSpaced(A.size(), 1, static_cast<Scalar>(A.size()));
  std::cout << "Matrix values:\n" << A << std::endl;
  return A;
}

template <typename Scalar>
Eigen::Matrix<Scalar, 1, Eigen::Dynamic> generateColMajorVector(
    const Eigen::DenseIndex size) {
  typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> ColMajorVector;
  ColMajorVector A(size);
  A.setLinSpaced(size, 1, static_cast<Scalar>(size));
  std::cout << "Vector values: " << A.transpose() << std::endl;
  return A;
}

template <typename Matrix, typename ReturnMatrix>
ReturnMatrix copy(const Eigen::MatrixBase<Matrix>& mat) {
  return mat;
}

template <typename Matrix>
Matrix copy_same(const Eigen::MatrixBase<Matrix>& mat) {
  return mat;
}

BOOST_PYTHON_MODULE(matrix) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  // Square matrix
  typedef Eigen::Matrix<double, 6, 6> Matrix6;
  eigenpy::enableEigenPySpecific<Matrix6>();

  // Non-square matrix
  typedef Eigen::Matrix<double, 4, 6> Matrix46;
  eigenpy::enableEigenPySpecific<Matrix46>();

  Eigen::MatrixXd (*naturalsXX)(int, int, bool) = naturals;
  Eigen::VectorXd (*naturalsX)(int, bool) = naturals;
  Eigen::Matrix3d (*naturals33)(bool) = naturals;

  bp::def("vector1x1", vector1x1<double>);
  bp::def("matrix1x1", matrix1x1<double>);
  bp::def("matrix1x1", matrix1x1_input<double>);
  bp::def("matrix1x1_int", matrix1x1_input<int>);

  bp::def("naturals", naturalsXX);
  bp::def("naturalsX", naturalsX);
  bp::def("naturals33", naturals33);

  bp::def("reflex", reflex<Eigen::MatrixXd>);
  bp::def("reflexV", reflex<Eigen::VectorXd>);
  bp::def("reflex33", reflex<Eigen::Matrix3d>);
  bp::def("reflex3", reflex<Eigen::Vector3d>);

  bp::def("emptyVector", emptyVector);
  bp::def("emptyMatrix", emptyMatrix);

  bp::def("base", base<VectorXd>);
  bp::def("base", base<MatrixXd>);

  bp::def("plain", plain<VectorXd>);
  bp::def("plain", plain<MatrixXd>);

  bp::def("matrix6", matrix6<double>);

  bp::def("generateRowMajorMatrix", generateRowMajorMatrix<double>);
  bp::def("generateRowMajorVector", generateRowMajorVector<double>);

  bp::def("generateColMajorMatrix", generateColMajorMatrix<double>);
  bp::def("generateColMajorVector", generateColMajorVector<double>);

  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMajorMatrixXd;
  bp::def("asRowMajorFromColMajorMatrix",
          copy<Eigen::MatrixXd, RowMajorMatrixXd>);
  bp::def("asRowMajorFromColMajorVector",
          copy<Eigen::VectorXd, Eigen::RowVectorXd>);
  bp::def("asRowMajorFromRowMajorMatrix",
          copy<RowMajorMatrixXd, RowMajorMatrixXd>);
  bp::def("asRowMajorFromRowMajorVector",
          copy<Eigen::RowVectorXd, Eigen::RowVectorXd>);

  bp::def("copyBoolToBool", copy_same<Eigen::Matrix<bool, -1, -1>>);

  bp::def("copyInt8ToInt8", copy_same<Eigen::Matrix<int8_t, -1, -1>>);
  bp::def("copyCharToChar", copy_same<Eigen::Matrix<char, -1, -1>>);
  bp::def("copyUCharToUChar", copy_same<Eigen::Matrix<unsigned char, -1, -1>>);

  bp::def("copyInt16ToInt16", copy_same<Eigen::Matrix<int16_t, -1, -1>>);
  bp::def("copyUInt16ToUInt16", copy_same<Eigen::Matrix<uint16_t, -1, -1>>);

  bp::def("copyInt32ToInt32", copy_same<Eigen::Matrix<int32_t, -1, -1>>);
  bp::def("copyUInt32ToUInt32", copy_same<Eigen::Matrix<uint32_t, -1, -1>>);

  bp::def("copyInt64ToInt64", copy_same<Eigen::Matrix<int64_t, -1, -1>>);
  bp::def("copyUInt64ToUInt64", copy_same<Eigen::Matrix<uint64_t, -1, -1>>);

  bp::def("copyLongToLong", copy_same<Eigen::Matrix<long, -1, -1>>);
  bp::def("copyULongToULong", copy_same<Eigen::Matrix<unsigned long, -1, -1>>);

  bp::def("copyLongLongToLongLong",
          copy_same<Eigen::Matrix<long long, -1, -1>>);
  bp::def("copyULongLongToULongLong",
          copy_same<Eigen::Matrix<unsigned long long, -1, -1>>);

  bp::def("copyFloatToFloat", copy_same<Eigen::Matrix<float, -1, -1>>);
  bp::def("copyDoubleToDouble", copy_same<Eigen::Matrix<double, -1, -1>>);
  bp::def("copyLongDoubleToLongDouble",
          copy_same<Eigen::Matrix<long double, -1, -1>>);

  bp::def("copyCFloatToCFloat",
          copy_same<Eigen::Matrix<std::complex<float>, -1, -1>>);
  bp::def("copyCDoubleToCDouble",
          copy_same<Eigen::Matrix<std::complex<double>, -1, -1>>);
  bp::def("copyCLongDoubleToCLongDouble",
          copy_same<Eigen::Matrix<std::complex<long double>, -1, -1>>);
}
