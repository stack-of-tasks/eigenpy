/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/user-type.hpp"
#include "eigenpy/ufunc.hpp"

#include <iostream>
#include <sstream>

template<typename Scalar>
struct CustomType
{
  CustomType() {}
  
  explicit CustomType(const Scalar & value)
  : m_value(value)
  {}
  
  CustomType operator*(const CustomType & other) const { return CustomType(m_value * other.m_value); }
  CustomType operator+(const CustomType & other) const { return CustomType(m_value + other.m_value); }
  CustomType operator-(const CustomType & other) const { return CustomType(m_value - other.m_value); }
  CustomType operator/(const CustomType & other) const { return CustomType(m_value / other.m_value); }
  
  void operator+=(const CustomType & other) { m_value += other.m_value; }
  void operator-=(const CustomType & other) { m_value -= other.m_value; }
  void operator*=(const CustomType & other) { m_value *= other.m_value; }
  void operator/=(const CustomType & other) { m_value /= other.m_value; }
  
  void operator=(const Scalar & value) { m_value = value; }
  
  bool operator==(const CustomType & other) const { return m_value == other.m_value; }
  bool operator!=(const CustomType & other) const { return m_value != other.m_value; }
  
  bool operator<=(const CustomType & other) const { return m_value <= other.m_value; }
  bool operator<(const CustomType & other) const { return m_value < other.m_value; }
  bool operator>=(const CustomType & other) const { return m_value >= other.m_value; }
  bool operator>(const CustomType & other) const { return m_value > other.m_value; }
  
  CustomType operator-() const { return CustomType(-m_value); }
  
  std::string print() const
  {
    std::stringstream ss;
    ss << "value: " << m_value << std::endl;
    return ss.str();
  }
 
protected:
  
  Scalar m_value;
};

template<typename Scalar>
Eigen::Matrix<CustomType<Scalar>,Eigen::Dynamic,Eigen::Dynamic> create(int rows, int cols)
{
  typedef Eigen::Matrix<CustomType<Scalar>,Eigen::Dynamic,Eigen::Dynamic> Matrix;
  return Matrix(rows,cols);
}

template<typename Scalar>
Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> build_matrix(int rows, int cols)
{
  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> Matrix;
  return Matrix(rows,cols);
}

template<typename Scalar>
void expose_custom_type(const std::string & name)
{
  using namespace Eigen;
  namespace bp = boost::python;
  
  typedef CustomType<Scalar> Type;
  
  bp::class_<Type>(name.c_str(),bp::init<Scalar>(bp::arg("value")))
  
  .def(bp::self + bp::self)
  .def(bp::self - bp::self)
  .def(bp::self * bp::self)
  .def(bp::self / bp::self)
  
  .def(bp::self += bp::self)
  .def(bp::self -= bp::self)
  .def(bp::self *= bp::self)
  .def(bp::self /= bp::self)
  
  .def("__repr__",&Type::print)
  ;
  
  int code = eigenpy::registerNewType<Type>();
  std::cout << "code: " << code << std::endl;
  eigenpy::registerCommonUfunc<Type>();
}

BOOST_PYTHON_MODULE(user_type)
{
  using namespace Eigen;
  namespace bp = boost::python;
  eigenpy::enableEigenPy();
  
  expose_custom_type<double>("CustomDouble");
  typedef CustomType<double> DoubleType;
  typedef Eigen::Matrix<DoubleType,Eigen::Dynamic,Eigen::Dynamic> DoubleMatrix;
  eigenpy::EigenToPyConverter<DoubleMatrix>::registration();
  bp::def("create_double",create<double>);
  
  expose_custom_type<float>("CustomFloat");
  typedef CustomType<float> FloatType;
  typedef Eigen::Matrix<FloatType,Eigen::Dynamic,Eigen::Dynamic> FloatMatrix;
  eigenpy::EigenToPyConverter<FloatMatrix>::registration();
  bp::def("create_float",create<float>);
  
  bp::def("build_matrix",build_matrix<double>);
}
