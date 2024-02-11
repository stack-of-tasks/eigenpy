/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/fwd.hpp"
#include "eigenpy/decompositions/decompositions.hpp"

#include "eigenpy/decompositions/sparse/cholmod/CholmodSimplicialLLT.hpp"
#include "eigenpy/decompositions/sparse/cholmod/CholmodSimplicialLDLT.hpp"
#include "eigenpy/decompositions/sparse/cholmod/CholmodSupernodalLLT.hpp"

namespace eigenpy {

void exposeCholmod() {
  using namespace Eigen;

  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> ColMajorSparseMatrix;
  //  typedef Eigen::SparseMatrix<double,Eigen::RowMajor> RowMajorSparseMatrix;

  bp::enum_<CholmodMode>("CholmodMode")
      .value("CholmodAuto", CholmodAuto)
      .value("CholmodSimplicialLLt", CholmodSimplicialLLt)
      .value("CholmodSupernodalLLt", CholmodSupernodalLLt)
      .value("CholmodLDLt", CholmodLDLt);

  CholmodSimplicialLLTVisitor<ColMajorSparseMatrix>::expose(
      "CholmodSimplicialLLT");
  CholmodSimplicialLDLTVisitor<ColMajorSparseMatrix>::expose(
      "CholmodSimplicialLDLT");
  CholmodSupernodalLLTVisitor<ColMajorSparseMatrix>::expose(
      "CholmodSupernodalLLT");
}
}  // namespace eigenpy
