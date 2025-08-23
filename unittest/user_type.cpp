/*
 * Copyright 2020 INRIA
 */

#include <iostream>
#include <sstream>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/ufunc.hpp"
#include "eigenpy/user-type.hpp"

template <typename Scalar>
struct CustomType;

namespace Eigen {
/// @brief Eigen::NumTraits<> specialization for casadi::SX
///
template <typename Scalar>
struct NumTraits<CustomType<Scalar>> {
  typedef CustomType<Scalar> Real;
  typedef CustomType<Scalar> NonInteger;
  typedef CustomType<Scalar> Literal;
  typedef CustomType<Scalar> Nested;

  enum {
    // does not support complex Base types
    IsComplex = 0,
    // does not support integer Base types
    IsInteger = 0,
    // only support signed Base types
    IsSigned = 1,
    // must initialize an AD<Base> object
    RequireInitialization = 1,
    // computational cost of the corresponding operations
    ReadCost = 1,
    AddCost = 2,
    MulCost = 2
  };

  static CustomType<Scalar> epsilon() {
    return CustomType<Scalar>(std::numeric_limits<Scalar>::epsilon());
  }

  static CustomType<Scalar> dummy_precision() {
    return CustomType<Scalar>(NumTraits<Scalar>::dummy_precision());
  }

  static CustomType<Scalar> highest() {
    return CustomType<Scalar>(std::numeric_limits<Scalar>::max());
  }

  static CustomType<Scalar> lowest() {
    return CustomType<Scalar>(std::numeric_limits<Scalar>::min());
  }

  static int digits10() { return std::numeric_limits<Scalar>::digits10; }
  static int max_digits10() {
    return std::numeric_limits<Scalar>::max_digits10;
  }
};
}  // namespace Eigen

template <typename Scalar>
struct CustomType {
  CustomType() {}

  explicit CustomType(const Scalar& value) : m_value(value) {}

  CustomType operator*(const CustomType& other) const {
    return CustomType(m_value * other.m_value);
  }
  CustomType operator+(const CustomType& other) const {
    return CustomType(m_value + other.m_value);
  }
  CustomType operator-(const CustomType& other) const {
    return CustomType(m_value - other.m_value);
  }
  CustomType operator/(const CustomType& other) const {
    return CustomType(m_value / other.m_value);
  }

  void operator+=(const CustomType& other) { m_value += other.m_value; }
  void operator-=(const CustomType& other) { m_value -= other.m_value; }
  void operator*=(const CustomType& other) { m_value *= other.m_value; }
  void operator/=(const CustomType& other) { m_value /= other.m_value; }

  void operator=(const Scalar& value) { m_value = value; }

  bool operator==(const CustomType& other) const {
    return m_value == other.m_value;
  }
  bool operator!=(const CustomType& other) const {
    return m_value != other.m_value;
  }

  bool operator<=(const CustomType& other) const {
    return m_value <= other.m_value;
  }
  bool operator<(const CustomType& other) const {
    return m_value < other.m_value;
  }
  bool operator>=(const CustomType& other) const {
    return m_value >= other.m_value;
  }
  bool operator>(const CustomType& other) const {
    return m_value > other.m_value;
  }

  CustomType operator-() const { return CustomType(-m_value); }

  operator Scalar() const { return m_value; }

  std::string print() const {
    std::stringstream ss;
    ss << "value: " << m_value << std::endl;
    return ss.str();
  }

  friend std::ostream& operator<<(std::ostream& os, const CustomType& X) {
    os << X.m_value;
    return os;
  }

  // protected:

  Scalar m_value;
};

template <typename Scalar>
Eigen::Matrix<CustomType<Scalar>, Eigen::Dynamic, Eigen::Dynamic> create(
    int rows, int cols) {
  typedef Eigen::Matrix<CustomType<Scalar>, Eigen::Dynamic, Eigen::Dynamic>
      Matrix;
  return Matrix(rows, cols);
}

template <typename Scalar>
void print(const Eigen::Matrix<CustomType<Scalar>, Eigen::Dynamic,
                               Eigen::Dynamic>& mat) {
  std::cout << mat << std::endl;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> build_matrix(int rows,
                                                                   int cols) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  return Matrix(rows, cols);
}

template <typename Scalar>
void expose_custom_type(const std::string& name) {
  using namespace Eigen;
  using eigenpy::literals::operator"" _a;
  namespace bp = boost::python;

  typedef CustomType<Scalar> Type;

  // use ""_a literal
  bp::class_<Type>(name.c_str(), bp::init<Scalar>("value"_a))
      .def(bp::self + bp::self)
      .def(bp::self - bp::self)
      .def(bp::self * bp::self)
      .def(bp::self / bp::self)

      .def(bp::self += bp::self)
      .def(bp::self -= bp::self)
      .def(bp::self *= bp::self)
      .def(bp::self /= bp::self)

      .def("__repr__", &Type::print);

  int code = eigenpy::registerNewType<Type>();
  std::cout << "code: " << code << std::endl;
  eigenpy::registerCommonUfunc<Type>();
}

BOOST_PYTHON_MODULE(user_type) {
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  expose_custom_type<double>("CustomDouble");
  typedef CustomType<double> DoubleType;
  typedef Eigen::Matrix<DoubleType, Eigen::Dynamic, Eigen::Dynamic>
      DoubleMatrix;
  eigenpy::EigenToPyConverter<DoubleMatrix>::registration();
  eigenpy::EigenFromPyConverter<DoubleMatrix>::registration();
  bp::def("create_double", create<double>);

  expose_custom_type<float>("CustomFloat");
  typedef CustomType<float> FloatType;
  typedef Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> FloatMatrix;
  eigenpy::EigenToPyConverter<FloatMatrix>::registration();
  eigenpy::EigenFromPyConverter<FloatMatrix>::registration();
  bp::def("create_float", create<float>);

  bp::def("build_matrix", build_matrix<double>);
#if EIGEN_VERSION_AT_LEAST(3, 3, 0)
  bp::def("print", print<double>);
  bp::def("print", print<float>);
#endif

  eigenpy::registerCast<DoubleType, double>(true);
  eigenpy::registerCast<double, DoubleType>(true);
  eigenpy::registerCast<DoubleType, float>(false);
  eigenpy::registerCast<float, DoubleType>(true);
  eigenpy::registerCast<DoubleType, int>(false);
  eigenpy::registerCast<int, DoubleType>(true);
  eigenpy::registerCast<DoubleType, long long>(false);
  eigenpy::registerCast<long long, DoubleType>(true);
  eigenpy::registerCast<DoubleType, long>(false);
  eigenpy::registerCast<long, DoubleType>(true);

  eigenpy::registerCast<FloatType, double>(true);
  eigenpy::registerCast<double, FloatType>(false);
  eigenpy::registerCast<FloatType, float>(true);
  eigenpy::registerCast<float, FloatType>(true);
  eigenpy::registerCast<FloatType, long long>(false);
  eigenpy::registerCast<long long, FloatType>(true);
  eigenpy::registerCast<FloatType, int>(false);
  eigenpy::registerCast<int, FloatType>(true);
  eigenpy::registerCast<FloatType, long>(false);
  eigenpy::registerCast<long, FloatType>(true);

  bp::implicitly_convertible<double, DoubleType>();
  bp::implicitly_convertible<DoubleType, double>();
}
