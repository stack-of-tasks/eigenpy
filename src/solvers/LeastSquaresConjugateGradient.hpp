/*
 * Copyright 2017, Justin Carpentier, LAAS-CNRS
 *
 * This file is part of eigenpy.
 * eigenpy is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * eigenpy is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.  You should
 * have received a copy of the GNU Lesser General Public License along
 * with eigenpy.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __eigenpy_least_square_conjugate_gradient_hpp__
#define __eigenpy_least_square_conjugate_gradient_hpp__

#include <boost/python.hpp>
#include <Eigen/IterativeLinearSolvers>

#include "eigenpy/solvers/IterativeSolverBase.hpp"

namespace Eigen
{
  template <typename _Scalar>
  class LeastSquareDiagonalPreconditionerFix
  : public LeastSquareDiagonalPreconditioner<_Scalar>
  {
    typedef _Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef LeastSquareDiagonalPreconditioner<_Scalar> Base;
    using DiagonalPreconditioner<_Scalar>::m_invdiag;
  public:
    
    LeastSquareDiagonalPreconditionerFix() : Base() {}
    
    template<typename MatType>
    explicit LeastSquareDiagonalPreconditionerFix(const MatType& mat) : Base()
    {
      compute(mat);
    }
    
    template<typename MatType>
    LeastSquareDiagonalPreconditionerFix& analyzePattern(const MatType& )
    {
      return *this;
    }
    
    template<typename MatType>
    LeastSquareDiagonalPreconditionerFix& factorize(const MatType& mat)
    {
      // Compute the inverse squared-norm of each column of mat
      m_invdiag.resize(mat.cols());
      if(MatType::IsRowMajor)
      {
        m_invdiag.setZero();
        for(Index j=0; j<mat.outerSize(); ++j)
        {
          for(typename MatType::InnerIterator it(mat,j); it; ++it)
            m_invdiag(it.index()) += numext::abs2(it.value());
        }
        for(Index j=0; j<mat.cols(); ++j)
          if(numext::real(m_invdiag(j))>RealScalar(0))
            m_invdiag(j) = RealScalar(1)/numext::real(m_invdiag(j));
      }
      else
      {
        for(Index j=0; j<mat.outerSize(); ++j)
        {
          RealScalar sum = mat.col(j).squaredNorm();
          if(sum>RealScalar(0))
            m_invdiag(j) = RealScalar(1)/sum;
          else
            m_invdiag(j) = RealScalar(1);
        }
      }
      Base::m_isInitialized = true;
      return *this;
    }
    
  };
}

namespace eigenpy
{
  
  namespace bp = boost::python;
  
  template<typename LeastSquaresConjugateGradient>
  struct LeastSquaresConjugateGradientVisitor
  : public boost::python::def_visitor< LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient> >
  {
    typedef Eigen::MatrixXd MatrixType;
    
    template<class PyClass>
    void visit(PyClass& cl) const
    {
      cl
      .def(bp::init<>("Default constructor"))
      .def(bp::init<MatrixType>(bp::arg("A"),"Initialize the solver with matrix A for further || Ax - b || solving.\n"
                                "This constructor is a shortcut for the default constructor followed by a call to compute()."))
      ;
      
    }
    
    static void expose()
    {
      bp::class_<LeastSquaresConjugateGradient,boost::noncopyable>("LeastSquaresConjugateGradient",
                                                       bp::no_init)
      .def(IterativeSolverVisitor<LeastSquaresConjugateGradient>())
      .def(LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient>())
      ;
      
    }
    
  };
  
} // namespace eigenpy

#endif // ifndef __eigenpy_least_square_conjugate_gradient_hpp__
