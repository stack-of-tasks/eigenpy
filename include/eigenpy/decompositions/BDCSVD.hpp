/*
 * Copyright 2025 INRIA
 */

#ifndef __eigenpy_decompositions_bdcsvd_hpp__
#define __eigenpy_decompositions_bdcsvd_hpp__

#include <Eigen/SVD>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"
#include "eigenpy/eigen/EigenBase.hpp"
#include "eigenpy/decompositions/SVDBase.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct BDCSVDVisitor
    : public boost::python::def_visitor<BDCSVDVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef Eigen::BDCSVD<MatrixType> Solver;
  typedef typename MatrixType::Scalar Scalar;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<Eigen::DenseIndex, Eigen::DenseIndex,
                      bp::optional<unsigned int>>(
            bp::args("self", "rows", "cols", "computationOptions "),
            "Default Constructor with memory preallocation. "))
        .def(bp::init<MatrixType, bp::optional<unsigned int>>(
            bp::args("self", "matrix", "computationOptions "),
            "Constructor performing the decomposition of given matrix."))

        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of columns. ")
        .def("compute",
             (Solver & (Solver::*)(const MatrixType &matrix)) & Solver::compute,
             bp::args("self", "matrix"),
             "Method performing the decomposition of given matrix. Computes "
             "Thin/Full "
             "unitaries U/V if specified using the Options template parameter "
             "or the class constructor. ",
             bp::return_self<>())
        .def("compute",
             (Solver & (Solver::*)(const MatrixType &matrix,
                                   unsigned int computationOptions)) &
                 Solver::compute,
             bp::args("self", "matrix", "computationOptions"),
             "Method performing the decomposition of given matrix, as "
             "specified by the computationOptions parameter.  ",
             bp::return_self<>())
        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows. ")
        .def("setSwitchSize", &Solver::setSwitchSize, bp::args("self", "s"))

        .def(SVDBaseVisitor<Solver>());
  }

  static void expose() {
    static const std::string classname =
        "BDCSVD_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "Class Bidiagonal Divide and Conquer SVD.\n\n"
        "This class first reduces the input matrix to bi-diagonal form using "
        "class "
        "UpperBidiagonalization, and then performs a divide-and-conquer "
        "diagonalization. "
        "Small blocks are diagonalized using class JacobiSVD. You can control "
        "the "
        "switching size with the setSwitchSize() method, default is 16. For "
        "small matrice "
        "(<16), it is thus preferable to directly use JacobiSVD. For larger "
        "ones, BDCSVD "
        "is highly recommended and can several order of magnitude faster.\n\n"
        "Warming: this algorithm is unlikely to provide accurate result when "
        "compiled with "
        "unsafe math optimizations. For instance, this concerns Intel's "
        "compiler (ICC), which "
        "performs such optimization by default unless you compile with the "
        "-fp-model precise "
        "option. Likewise, the -ffast-math option of GCC or clang will "
        "significantly degrade the "
        "accuracy.",
        bp::no_init)
        .def(BDCSVDVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_bdcsvd_hpp__
