/*
 * Copyright 2022, CNRS
 * Copyright 2022, INRIA
 */

#include "eigenpy/std-vector.hpp"

namespace eigenpy {

void exposeStdVector() {
  exposeStdVectorEigenSpecificType<Eigen::MatrixXd>("MatrixXd");
  exposeStdVectorEigenSpecificType<Eigen::VectorXd>("VectorXd");

  exposeStdVectorEigenSpecificType<Eigen::MatrixXi>("MatrixXi");
  exposeStdVectorEigenSpecificType<Eigen::VectorXi>("VectorXi");
}

}  // namespace eigenpy
