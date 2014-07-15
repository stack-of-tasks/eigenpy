#include "../src/eigenpy.hpp"

Eigen::MatrixXd naturals(int R,int C,bool verbose)
{
  Eigen::MatrixXd mat(R,C);
  for(int r=0;r<R;++r)
    for(int c=0;c<C;++c)
      mat(r,c) = r*C+c;

  if(verbose)
    std::cout << "EigenMat = " << mat << std::endl;
  return mat;
}

Eigen::VectorXd naturals(int R,bool verbose)
{
  Eigen::VectorXd mat(R);
  for(int r=0;r<R;++r)
    mat[r] = r;

  if(verbose)
    std::cout << "EigenMat = " << mat << std::endl;
  return mat;
}

Eigen::Matrix3d naturals(bool verbose)
{
  Eigen::Matrix3d mat;
  for(int r=0;r<3;++r)
    for(int c=0;c<3;++c)
      mat(r,c) = r*3+c;

  if(verbose)
    std::cout << "EigenMat = " << mat << std::endl;
  return mat;
}

template<typename MatType>
Eigen::MatrixXd reflex(const MatType & M, bool verbose)
{
  if(verbose)
    std::cout << "EigenMat = " << M << std::endl;
  return Eigen::MatrixXd(M);
}

BOOST_PYTHON_MODULE(libeigenpy)
{
  namespace bp = boost::python;
  eigenpy::enableEigenPy();

  Eigen::MatrixXd (*naturalsXX)(int,int,bool) = naturals;
  Eigen::VectorXd (*naturalsX)(int,bool) = naturals;
  Eigen::Matrix3d (*naturals33)(bool) = naturals;

  bp::def("naturals", naturalsXX);
  bp::def("naturalsX", naturalsX);
  bp::def("naturals33", naturals33);

  bp::def("reflex", reflex<Eigen::MatrixXd>);
  bp::def("reflexV", reflex<Eigen::VectorXd>);
  bp::def("reflex33", reflex<Eigen::Matrix3d>);
}
