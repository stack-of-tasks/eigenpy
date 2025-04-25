/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"

namespace Eigen {
#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)    \
  /** \ingroup matrixtypedefs */                                   \
  typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix; \
  /** \ingroup matrixtypedefs */                                   \
  typedef Matrix<Type, Size, 1> Vector##SizeSuffix##TypeSuffix;    \
  /** \ingroup matrixtypedefs */                                   \
  typedef Matrix<Type, 1, Size> RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)          \
  /** \ingroup matrixtypedefs */                                   \
  typedef Matrix<Type, Size, Dynamic> Matrix##Size##X##TypeSuffix; \
  /** \ingroup matrixtypedefs */                                   \
  typedef Matrix<Type, Dynamic, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2)           \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3)           \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4)           \
  EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X)     \
  EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2)        \
  EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3)        \
  EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

EIGEN_MAKE_TYPEDEFS_ALL_SIZES(long double, ld)
EIGEN_MAKE_TYPEDEFS_ALL_SIZES(std::complex<long double>, cld)

#undef EIGEN_MAKE_TYPEDEFS_ALL_SIZES
#undef EIGEN_MAKE_TYPEDEFS
#undef EIGEN_MAKE_FIXED_TYPEDEFS
}  // namespace Eigen

template <typename ComplexMatrix>
typename Eigen::Matrix<typename ComplexMatrix::RealScalar,
                       ComplexMatrix::RowsAtCompileTime,
                       ComplexMatrix::ColsAtCompileTime, ComplexMatrix::Options>
real(const Eigen::MatrixBase<ComplexMatrix> &complex_mat) {
  return complex_mat.real();
}

template <typename ComplexMatrix>
typename Eigen::Matrix<typename ComplexMatrix::RealScalar,
                       ComplexMatrix::RowsAtCompileTime,
                       ComplexMatrix::ColsAtCompileTime, ComplexMatrix::Options>
imag(const Eigen::MatrixBase<ComplexMatrix> &complex_mat) {
  return complex_mat.imag();
}

template <typename Scalar, int Rows, int Cols, int Options>
Eigen::Matrix<std::complex<Scalar>, Rows, Cols, Options> ascomplex(
    const Eigen::Matrix<Scalar, Rows, Cols, Options> &mat) {
  typedef Eigen::Matrix<std::complex<Scalar>, Rows, Cols, Options> ReturnType;
  return ReturnType(mat.template cast<std::complex<Scalar>>());
}

BOOST_PYTHON_MODULE(complex) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  bp::def("ascomplex", ascomplex<float, Eigen::Dynamic, Eigen::Dynamic, 0>);
  bp::def("ascomplex", ascomplex<double, Eigen::Dynamic, Eigen::Dynamic, 0>);
  bp::def("ascomplex",
          ascomplex<long double, Eigen::Dynamic, Eigen::Dynamic, 0>);

  bp::def("real",
          (MatrixXf (*)(const Eigen::MatrixBase<MatrixXcf> &))&real<MatrixXcf>);
  bp::def("real",
          (MatrixXd (*)(const Eigen::MatrixBase<MatrixXcd> &))&real<MatrixXcd>);
  bp::def(
      "real",
      (MatrixXld (*)(const Eigen::MatrixBase<MatrixXcld> &))&real<MatrixXcld>);

  bp::def("imag",
          (MatrixXf (*)(const Eigen::MatrixBase<MatrixXcf> &))&imag<MatrixXcf>);
  bp::def("imag",
          (MatrixXd (*)(const Eigen::MatrixBase<MatrixXcd> &))&imag<MatrixXcd>);
  bp::def(
      "imag",
      (MatrixXld (*)(const Eigen::MatrixBase<MatrixXcld> &))&imag<MatrixXcld>);
}
