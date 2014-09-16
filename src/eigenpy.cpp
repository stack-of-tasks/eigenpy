#include "eigenpy/eigenpy.hpp"

namespace eigenpy
{

  /* Enable Eigen-Numpy serialization for a set of standard MatrixBase instance. */
  void enableEigenPy()
  {
    Exception::registerException();

    enableEigenPySpecific<Eigen::MatrixXd,Eigen::MatrixXd>();
    enableEigenPySpecific<Eigen::Matrix2d,Eigen::Matrix2d>();
    enableEigenPySpecific<Eigen::Matrix3d,Eigen::Matrix3d>();
    enableEigenPySpecific<Eigen::Matrix4d,Eigen::Matrix4d>();

    enableEigenPySpecific<Eigen::VectorXd,Eigen::VectorXd>();
    enableEigenPySpecific<Eigen::Vector2d,Eigen::Vector2d>();
    enableEigenPySpecific<Eigen::Vector3d,Eigen::Vector3d>();
    enableEigenPySpecific<Eigen::Vector4d,Eigen::Vector4d>();
  }

} // namespace eigenpy
