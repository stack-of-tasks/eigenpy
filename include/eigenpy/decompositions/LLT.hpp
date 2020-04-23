/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_decomposition_llt_hpp__
#define __eigenpy_decomposition_llt_hpp__

#include "eigenpy/eigenpy.hpp"

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy
{
  
  template<typename _MatrixType>
  struct LLTSolverVisitor
  : public boost::python::def_visitor< LLTSolverVisitor<_MatrixType> >
  {
    
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1,MatrixType::Options> VectorType;
    typedef Eigen::LLT<MatrixType> Solver;
    
    template<class PyClass>
    void visit(PyClass& cl) const
    {
      namespace bp = boost::python;
      cl
      .def(bp::init<>("Default constructor"))
      .def(bp::init<Eigen::DenseIndex>(bp::arg("size"),
                                       "Default constructor with memory preallocation"))
      .def(bp::init<MatrixType>(bp::arg("matrix"),
                                "Constructs a LLT factorization from a given matrix."))
       
      .def("matrixL",&matrixL,bp::arg("self"),
           "Returns the lower triangular matrix L.")
      .def("matrixU",&matrixU,bp::arg("self"),
           "Returns the upper triangular matrix U.")
      .def("matrixLLT",&Solver::matrixLLT,bp::arg("self"),
           "Returns the LLT decomposition matrix.",
           bp::return_internal_reference<>())

#if EIGEN_VERSION_AT_LEAST(3,3,90)
      .def("rankUpdate",(Solver& (Solver::*)(const VectorType &, const RealScalar &))&Solver::template rankUpdate<VectorType>,
           bp::args("self","vector","sigma"), bp::return_self<>())
#else
      .def("rankUpdate",(Solver (Solver::*)(const VectorType &, const RealScalar &))&Solver::template rankUpdate<VectorType>,
           bp::args("self","vector","sigma"))
#endif
      
#if EIGEN_VERSION_AT_LEAST(3,3,0)
      .def("adjoint",&Solver::adjoint,bp::arg("self"),
           "Returns the adjoint, that is, a reference to the decomposition itself as if the underlying matrix is self-adjoint.",
           bp::return_self<>())
#endif
      
      .def("compute",(Solver & (Solver::*)(const Eigen::EigenBase<MatrixType> & matrix))&Solver::compute,
           bp::args("self","matrix"),
           "Computes the LLT of given matrix.",
           bp::return_self<>())
      
      .def("info",&Solver::info,bp::arg("self"),
           "NumericalIssue if the input contains INF or NaN values or overflow occured. Returns Success otherwise.")
#if EIGEN_VERSION_AT_LEAST(3,3,0)
      .def("rcond",&Solver::rcond,bp::arg("self"),
           "Returns an estimate of the reciprocal condition number of the matrix.")
#endif
      .def("reconstructedMatrix",&Solver::reconstructedMatrix,bp::arg("self"),
           "Returns the matrix represented by the decomposition, i.e., it returns the product: L L^*. This function is provided for debug purpose.")
      .def("solve",&solve<VectorType>,bp::args("self","b"),
           "Returns the solution x of A x = b using the current decomposition of A.")
      ;
    }
    
    static void expose()
    {
      static const std::string classname = "LLT" + scalar_name<Scalar>::shortname();
      expose(classname);
    }
    
    static void expose(const std::string & name)
    {
      namespace bp = boost::python;
      bp::class_<Solver>(name.c_str(),
                         "Standard Cholesky decomposition (LL^T) of a matrix and associated features.\n\n"
                         "This class performs a LL^T Cholesky decomposition of a symmetric, positive definite matrix A such that A = LL^* = U^*U, where L is lower triangular.\n\n"
                         "While the Cholesky decomposition is particularly useful to solve selfadjoint problems like D^*D x = b, for that purpose, we recommend the Cholesky decomposition without square root which is more stable and even faster. Nevertheless, this standard Cholesky decomposition remains useful in many other situations like generalised eigen problems with hermitian matrices.",
                         bp::no_init)
      .def(LLTSolverVisitor());
    }
    
  private:
    
    static MatrixType matrixL(const Solver & self) { return self.matrixL(); }
    static MatrixType matrixU(const Solver & self) { return self.matrixU(); }
    
    template<typename VectorType>
    static VectorType solve(const Solver & self, const VectorType & vec)
    {
      return self.solve(vec);
    }
  };
  
} // namespace eigenpy

#endif // ifndef __eigenpy_decomposition_llt_hpp__
