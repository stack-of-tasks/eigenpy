/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_permutation_matrix_hpp__
#define __eigenpy_decompositions_permutation_matrix_hpp__

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/eigen/EigenBase.hpp"

namespace eigenpy {

template <int SizeAtCompileTime, int MaxSizeAtCompileTime = SizeAtCompileTime,
          typename StorageIndex_ = int>
struct PermutationMatrixVisitor
    : public boost::python::def_visitor<PermutationMatrixVisitor<
          SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex_>> {
  typedef StorageIndex_ StorageIndex;
  typedef Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime,
                                   StorageIndex>
      PermutationMatrix;
  typedef typename PermutationMatrix::DenseMatrixType DenseMatrixType;
  typedef PermutationMatrix Self;
  typedef Eigen::Matrix<StorageIndex, SizeAtCompileTime, 1, 0,
                        MaxSizeAtCompileTime, 1>
      VectorIndex;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<const Eigen::DenseIndex>(bp::args("self", "size"),
                                             "Default constructor"))
        .def(bp::init<VectorIndex>(
            bp::args("self", "indices"),
            "The indices array has the meaning that the permutations sends "
            "each integer i to indices[i].\n"
            "It is your responsibility to check that the indices array that "
            "you passes actually describes a permutation, i.e., each value "
            "between 0 and n-1 occurs exactly once, where n is the array's "
            "size."))

        .def(
            "indices",
            +[](const PermutationMatrix &self) {
              return VectorIndex(self.indices());
            },
            bp::arg("self"), "The stored array representing the permutation.")

        .def("applyTranspositionOnTheLeft",
             &PermutationMatrix::applyTranspositionOnTheLeft,
             bp::args("self", "i", "j"),
             "Multiplies self by the transposition (ij) on the left.",
             bp::return_self<>())
        .def("applyTranspositionOnTheRight",
             &PermutationMatrix::applyTranspositionOnTheRight,
             bp::args("self", "i", "j"),
             "Multiplies self by the transposition (ij) on the right.",
             bp::return_self<>())

        .def("setIdentity",
             (void (PermutationMatrix::*)())&PermutationMatrix::setIdentity,
             bp::arg("self"),
             "Sets self to be the identity permutation matrix.")
        .def("setIdentity",
             (void (PermutationMatrix::*)(
                 Eigen::DenseIndex))&PermutationMatrix::setIdentity,
             bp::args("self", "size"),
             "Sets self to be the identity permutation matrix of given size.")

        .def("toDenseMatrix", &PermutationMatrix::toDenseMatrix,
             bp::arg("self"),
             "Returns a numpy array object initialized from this permutation "
             "matrix.")

        .def(
            "transpose",
            +[](const PermutationMatrix &self) -> PermutationMatrix {
              return self.transpose();
            },
            bp::arg("self"), "Returns the tranpose permutation matrix.")
        .def(
            "inverse",
            +[](const PermutationMatrix &self) -> PermutationMatrix {
              return self.inverse();
            },
            bp::arg("self"), "Returns the inverse permutation matrix.")

        .def("resize", &PermutationMatrix::resize, bp::args("self", "size"),
             "Resizes to given size.")

        .def(bp::self * bp::self)
        .def(EigenBaseVisitor<Self>());
  }

  static void expose(const std::string &name) {
    bp::class_<PermutationMatrix>(name.c_str(),
                                  "Permutation matrix.\n"
                                  "This class represents a permutation matrix, "
                                  "internally stored as a vector of integers.",
                                  bp::no_init)
        .def(IdVisitor<PermutationMatrix>())
        .def(PermutationMatrixVisitor());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_permutation_matrix_hpp__
