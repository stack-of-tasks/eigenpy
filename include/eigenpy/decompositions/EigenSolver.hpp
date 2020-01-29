/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_decomposition_eigen_solver_hpp__
#define __eigenpy_decomposition_eigen_solver_hpp__

#include <boost/python.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy
{
  
  template<typename _MatrixType>
  struct EigenSolverVisitor
  : public boost::python::def_visitor< EigenSolverVisitor<_MatrixType> >
  {
    
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef Eigen::EigenSolver<MatrixType> Solver;
    
    template<class PyClass>
    void visit(PyClass& cl) const
    {
      namespace bp = boost::python;
      cl
      .def(bp::init<>("Default constructor"))
      .def(bp::init<Eigen::Index>(bp::arg("size"),
                                  "Default constructor with memory preallocation"))
      .def(bp::init<MatrixType,bp::optional<bool> >(bp::args("matrix","compute_eigen_vectors"),
                                                    "Computes eigendecomposition of given matrix"))
       
      .def("eigenvalues",&Solver::eigenvalues,bp::arg("self"),
           "Returns the eigenvalues of given matrix.",
           bp::return_value_policy<bp::return_by_value>())
      .def("eigenvectors",&Solver::eigenvectors,bp::arg("self"),
           "Returns the eigenvectors of given matrix.",
           bp::return_value_policy<bp::return_by_value>())
      
      .def("compute",(Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> & matrix))&Solver::template compute<MatrixType>,
           bp::args("self","matrix"),
           "Computes the eigendecomposition of given matrix.",
           bp::return_value_policy<bp::reference_existing_object>())
      .def("compute",(Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> & matrix, bool))&Solver::compute,
           bp::args("self","matrix","compute_eigen_vectors"),
           "Computes theeigendecomposition of given matrix.",
           bp::return_value_policy<bp::reference_existing_object>())
      
      .def("getMaxIterations",&Solver::getMaxIterations,bp::arg("self"),
           "Returns the maximum number of iterations.")
      .def("setMaxIterations",&Solver::setMaxIterations,bp::args("self","max_iter"),
           "Sets the maximum number of iterations allowed.",
           bp::return_value_policy<bp::reference_existing_object>())
      
      .def("pseudoEigenvalueMatrix",&Solver::pseudoEigenvalueMatrix,bp::arg("self"),
           "Returns the block-diagonal matrix in the pseudo-eigendecomposition.")
      .def("pseudoEigenvectors",&Solver::pseudoEigenvectors ,bp::arg("self"),
           "Returns the pseudo-eigenvectors of given matrix.",
           bp::return_value_policy<bp::return_by_value>())
      
      .def("info",&Solver::info,bp::arg("self"),
           "NumericalIssue if the input contains INF or NaN values or overflow occured. Returns Success otherwise.")
      ;
    }
    
    static void expose()
    {
      static const std::string classname = "EigenSolver" + scalar_name<Scalar>::shortname();
      expose(classname);
    }
    
    static void expose(const std::string & name)
    {
      namespace bp = boost::python;
      bp::class_<Solver>(name.c_str(),
                         bp::no_init)
      .def(EigenSolverVisitor());
    }
    
  };
  
} // namespace eigenpy

#endif // ifndef __eigenpy_decomposition_eigen_solver_hpp__

