#include "eigenpy/quaternion.hpp"
#include "eigenpy/geometry.hpp"

namespace eigenpy
{
  void exposeQuaternion()
  {
      QuaternionVisitor<Eigen::Quaterniond>::expose();
  }
} // namespace eigenpy
