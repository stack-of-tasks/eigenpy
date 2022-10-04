/*
 * Copyright 2022, CNRS
 * Copyright 2022, INRIA
 */

#include "eigenpy/std-vector.hpp"

namespace eigenpy {

template <typename MatType>
void exposeStdVectorEigenSpecificType(const char* name) {
  std::string full_name = "StdVec_";
  full_name += name;
  StdVectorPythonVisitor<std::vector<MatType>, true>::expose(full_name.c_str());
}

void exposeStdVector() {
  exposeStdVectorEigenSpecificType<Eigen::MatrixXd>("MatrixXd");
  exposeStdVectorEigenSpecificType<Eigen::VectorXd>("VectorXd");

  exposeStdVectorEigenSpecificType<Eigen::MatrixXi>("MatrixXi");
  exposeStdVectorEigenSpecificType<Eigen::VectorXi>("VectorXi");
}

}  // namespace eigenpy
