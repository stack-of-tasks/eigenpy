#include "../src/eigenpy.hpp"

Eigen::MatrixXd test()
{
  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(3,6);
  std::cout << "EigenMAt = " << mat << std::endl;
  return mat;
}
Eigen::VectorXd testVec()
{
  Eigen::VectorXd mat = Eigen::VectorXd::Random(6);
  std::cout << "EigenVec = " << mat << std::endl;
  return mat;
}

void test2( Eigen::MatrixXd mat )
{
  std::cout << "Test2 mat = " << mat << std::endl;
}
void test2Vec( Eigen::VectorXd v )
{
  std::cout << "Test2 vec = " << v << std::endl;
}

BOOST_PYTHON_MODULE(libeigenpy)
{

  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  bp::def("test", test);
  bp::def("testVec", testVec);
  bp::def("test2", test2);
  bp::def("test2Vec", test2Vec);
}
