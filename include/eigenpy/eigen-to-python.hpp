//
// Copyright (c) 2014-2020 CNRS INRIA
//

#ifndef __eigenpy_eigen_to_python_hpp__
#define __eigenpy_eigen_to_python_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/eigen-allocator.hpp"

namespace eigenpy
{
  namespace bp = boost::python;

    template<typename MatType>
    struct EigenToPy
    {
      static PyObject* convert(MatType const & mat)
      {
        typedef typename MatType::Scalar Scalar;
        assert( (mat.rows()<INT_MAX) && (mat.cols()<INT_MAX)
          && "Matrix range larger than int ... should never happen." );
        const npy_intp R = (npy_intp)mat.rows(), C = (npy_intp)mat.cols();

        PyArrayObject* pyArray;
        // Allocate Python memory
        if( ( ((!(C == 1) != !(R == 1)) && !MatType::IsVectorAtCompileTime) || MatType::IsVectorAtCompileTime)
           && NumpyType::getType() == ARRAY_TYPE) // Handle array with a single dimension
        {
          npy_intp shape[1] = { C == 1 ? R : C };
          pyArray = (PyArrayObject*) PyArray_SimpleNew(1, shape,
                                                       NumpyEquivalentType<Scalar>::type_code);
        }
        else
        {
          npy_intp shape[2] = { R,C };
          pyArray = (PyArrayObject*) PyArray_SimpleNew(2, shape,
                                                       NumpyEquivalentType<Scalar>::type_code);
        }

        // Copy data
        EigenAllocator<MatType>::copy(mat,pyArray);
        
        // Create an instance (either np.array or np.matrix)
        return NumpyType::getInstance().make(pyArray).ptr();
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

#if EIGEN_VERSION_AT_LEAST(3,2,0)
  template<typename MatType>
  struct EigenToPyConverter< eigenpy::Ref<MatType> >
  {
    static void registration()
    {
    }
  };
#endif
}

#endif // __eigenpy_eigen_to_python_hpp__
