/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/JacobiSVD.hpp"

namespace eigenpy {
void exposeJacobiSVDSolver() {
  using namespace Eigen;
  using Eigen::JacobiSVD;

  using Eigen::ColPivHouseholderQRPreconditioner;
  using Eigen::FullPivHouseholderQRPreconditioner;
  using Eigen::HouseholderQRPreconditioner;
  using Eigen::NoQRPreconditioner;

  using ColPivHhJacobiSVD =
      JacobiSVD<MatrixXd, ColPivHouseholderQRPreconditioner>;
  using FullPivHhJacobiSVD =
      JacobiSVD<MatrixXd, FullPivHouseholderQRPreconditioner>;
  using HhJacobiSVD = JacobiSVD<MatrixXd, HouseholderQRPreconditioner>;
  using NoPrecondJacobiSVD = JacobiSVD<MatrixXd, NoQRPreconditioner>;

  JacobiSVDVisitor<ColPivHhJacobiSVD>::expose("ColPivHhJacobiSVD");
  JacobiSVDVisitor<FullPivHhJacobiSVD>::expose("FullPivHhJacobiSVD");
  JacobiSVDVisitor<HhJacobiSVD>::expose("HhJacobiSVD");
  JacobiSVDVisitor<NoPrecondJacobiSVD>::expose("NoPrecondJacobiSVD");
}
}  // namespace eigenpy
