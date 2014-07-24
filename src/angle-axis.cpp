#include "eigenpy/angle-axis.hpp"
#include "eigenpy/geometry.hpp"

namespace eigenpy
{
  void exposeAngleAxis()
  {
    AngleAxisVisitor<Eigen::AngleAxisd>::expose();
  }
} // namespace eigenpy
