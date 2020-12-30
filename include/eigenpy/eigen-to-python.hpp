//
// Copyright (c) 2014-2020 CNRS INRIA
//

#ifndef __eigenpy_eigen_to_python_hpp__
#define __eigenpy_eigen_to_python_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/numpy-allocator.hpp"

#include <boost/type_traits.hpp>

namespace boost { namespace python {

  template<typename MatrixRef, class MakeHolder>
  struct to_python_indirect_eigen
  {
    template <class U>
    inline PyObject* operator()(U const& mat) const
    {
      return eigenpy::EigenToPy<MatrixRef>::convert(const_cast<U&>(mat));
    }
    
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
    inline PyTypeObject const*
    get_pytype() const
    {
      return converter::registered_pytype<MatrixRef>::get_pytype();
    }
#endif
  };

  template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime, class MakeHolder>
  struct to_python_indirect<Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime,Options,MaxRowsAtCompileTime,MaxColsAtCompileTime>&,MakeHolder>
  : to_python_indirect_eigen<Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime,Options,MaxRowsAtCompileTime,MaxColsAtCompileTime>&,MakeHolder>
  {
  };

  template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime, int Options, int MaxRowsAtCompileTime, int MaxColsAtCompileTime, class MakeHolder>
  struct to_python_indirect<const Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime,Options,MaxRowsAtCompileTime,MaxColsAtCompileTime>&,MakeHolder>
  : to_python_indirect_eigen<const Eigen::Matrix<Scalar,RowsAtCompileTime,ColsAtCompileTime,Options,MaxRowsAtCompileTime,MaxColsAtCompileTime>&,MakeHolder>
  {
  };

}}

namespace eigenpy
{
  namespace bp = boost::python;

  template<typename MatType, typename _Scalar>
  struct EigenToPy
  {
    static PyObject* convert(typename boost::add_reference<typename boost::add_const<MatType>::type>::type mat)
    {
      typedef typename boost::remove_const<typename boost::remove_reference<MatType>::type>::type MatrixDerived;
      
      assert( (mat.rows()<INT_MAX) && (mat.cols()<INT_MAX)
             && "Matrix range larger than int ... should never happen." );
      const npy_intp R = (npy_intp)mat.rows(), C = (npy_intp)mat.cols();
      
      PyArrayObject* pyArray;
      // Allocate Python memory
      if( ( ((!(C == 1) != !(R == 1)) && !MatrixDerived::IsVectorAtCompileTime) || MatrixDerived::IsVectorAtCompileTime)
         && NumpyType::getType() == ARRAY_TYPE) // Handle array with a single dimension
      {
        npy_intp shape[1] = { C == 1 ? R : C };
        pyArray = NumpyAllocator<MatType>::allocate(const_cast<MatrixDerived &>(mat.derived()),
                                                    1,shape);
      }
      else
      {
        npy_intp shape[2] = { R,C };
        pyArray = NumpyAllocator<MatType>::allocate(const_cast<MatrixDerived &>(mat.derived()),
                                                    2,shape);
      }
      
      // Create an instance (either np.array or np.matrix)
      return NumpyType::make(pyArray).ptr();
    }
  };

  template<typename MatType, int Options, typename Stride, typename _Scalar>
  struct EigenToPy< Eigen::Ref<MatType,Options,Stride>,_Scalar >
  {
    static PyObject* convert(const Eigen::Ref<MatType,Options,Stride> & mat)
    {
      typedef Eigen::Ref<MatType,Options,Stride> EigenRef;
      
      assert( (mat.rows()<INT_MAX) && (mat.cols()<INT_MAX)
             && "Matrix range larger than int ... should never happen." );
      const npy_intp R = (npy_intp)mat.rows(), C = (npy_intp)mat.cols();
      
      PyArrayObject* pyArray;
      // Allocate Python memory
      if( ( ((!(C == 1) != !(R == 1)) && !MatType::IsVectorAtCompileTime) || MatType::IsVectorAtCompileTime)
         && NumpyType::getType() == ARRAY_TYPE) // Handle array with a single dimension
      {
        npy_intp shape[1] = { C == 1 ? R : C };
        pyArray = NumpyAllocator<EigenRef>::allocate(const_cast<EigenRef &>(mat),1,shape);
      }
      else
      {
        npy_intp shape[2] = { R,C };
        pyArray = NumpyAllocator<EigenRef>::allocate(const_cast<EigenRef &>(mat),2,shape);
      }
      
      // Create an instance (either np.array or np.matrix)
      return NumpyType::make(pyArray).ptr();
    }
  };

  template<typename MatType>
  struct EigenToPyConverter
  {
    static void registration()
    {
      bp::to_python_converter<MatType,EigenToPy<MatType> >();
    }
  };
}

#endif // __eigenpy_eigen_to_python_hpp__
