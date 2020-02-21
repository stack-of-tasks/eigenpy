/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2020, INRIA
 */

#ifndef __eigenpy_details_hpp__
#define __eigenpy_details_hpp__

#include "eigenpy/details/rvalue_from_python_data.hpp"
#include "eigenpy/fwd.hpp"

#include <patchlevel.h> // For PY_MAJOR_VERSION
#include <iostream>

#include "eigenpy/numpy-type.hpp"
#include "eigenpy/scalar-conversion.hpp"
#include "eigenpy/eigenpy.hpp"

#include "eigenpy/eigen-allocator.hpp"

#include "eigenpy/registration.hpp"
#include "eigenpy/map.hpp"
#include "eigenpy/exception.hpp"

#define GET_PY_ARRAY_TYPE(array) PyArray_ObjectType(reinterpret_cast<PyObject *>(array), 0)

namespace boost { namespace python { namespace detail {

  template<class MatType>
  struct referent_size<Eigen::MatrixBase<MatType>&>
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::MatrixBase<MatType> >
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::EigenBase<MatType>&>
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

  template<class MatType>
  struct referent_size<Eigen::EigenBase<MatType> >
  {
      BOOST_STATIC_CONSTANT(
          std::size_t, value = sizeof(MatType));
  };

}}}

namespace eigenpy
{

  namespace bp = boost::python;

  /* --- TO PYTHON -------------------------------------------------------------- */
  
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
          std::cout << "basic convert" << std::endl;
      if( ( ((!(C == 1) != !(R == 1)) && !MatType::IsVectorAtCompileTime) || MatType::IsVectorAtCompileTime)
         && NumpyType::getType() == ARRAY_TYPE) // Handle array with a single dimension
      {
          std::cout << "mat1\n" << mat << std::endl;
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

      // Allocate memory
      EigenObjectAllocator<MatType>::copy(mat,pyArray);
      
      // Create an instance (either np.array or np.matrix)
      return NumpyType::getInstance().make(pyArray).ptr();
    }
  };
  
  /* --- FROM PYTHON ------------------------------------------------------------ */

  template<typename MatType>
  struct EigenFromPy
  {
    
    static bool isScalarConvertible(const int np_type)
    {
      if(NumpyEquivalentType<typename MatType::Scalar>::type_code == np_type)
        return true;
      
      switch(np_type)
      {
        case NPY_INT:
          return FromTypeToType<int,typename MatType::Scalar>::value;
        case NPY_LONG:
          return FromTypeToType<long,typename MatType::Scalar>::value;
        case NPY_FLOAT:
          return FromTypeToType<float,typename MatType::Scalar>::value;
        case NPY_CFLOAT:
          return FromTypeToType<std::complex<float>,typename MatType::Scalar>::value;
        case NPY_DOUBLE:
          return FromTypeToType<double,typename MatType::Scalar>::value;
        case NPY_CDOUBLE:
          return FromTypeToType<std::complex<double>,typename MatType::Scalar>::value;
        case NPY_LONGDOUBLE:
          return FromTypeToType<long double,typename MatType::Scalar>::value;
        case NPY_CLONGDOUBLE:
          return FromTypeToType<std::complex<long double>,typename MatType::Scalar>::value;
        default:
          return false;
      }
    }
    
    /// \brief Determine if pyObj can be converted into a MatType object
    static void* convertible(PyArrayObject* pyArray)
    {
      if(!PyArray_Check(pyArray))
        return 0;
      
      if(!isScalarConvertible(GET_PY_ARRAY_TYPE(pyArray)))
        return 0;

      if(MatType::IsVectorAtCompileTime)
      {
        const Eigen::DenseIndex size_at_compile_time
        = MatType::IsRowMajor
        ? MatType::ColsAtCompileTime
        : MatType::RowsAtCompileTime;
        
        switch(PyArray_NDIM(pyArray))
        {
          case 0:
            return 0;
          case 1:
          {
            if(size_at_compile_time != Eigen::Dynamic)
            {
              // check that the sizes at compile time matche
              if(PyArray_DIMS(pyArray)[0] == size_at_compile_time)
                return pyArray;
              else
                return 0;
            }
            else // This is a dynamic MatType
              return pyArray;
          }
          case 2:
          {
            // Special care of scalar matrix of dimension 1x1.
            if(PyArray_DIMS(pyArray)[0] == 1 && PyArray_DIMS(pyArray)[1] == 1)
            {
              if(size_at_compile_time != Eigen::Dynamic)
              {
                if(size_at_compile_time == 1)
                  return pyArray;
                else
                  return 0;
              }
              else // This is a dynamic MatType
                return pyArray;
            }
            
            if(PyArray_DIMS(pyArray)[0] > 1 && PyArray_DIMS(pyArray)[1] > 1)
            {
#ifndef NDEBUG
              std::cerr << "The number of dimension of the object does not correspond to a vector" << std::endl;
#endif
              return 0;
            }
            
            if(((PyArray_DIMS(pyArray)[0] == 1) && (MatType::ColsAtCompileTime == 1))
               || ((PyArray_DIMS(pyArray)[1] == 1) && (MatType::RowsAtCompileTime == 1)))
            {
#ifndef NDEBUG
              if(MatType::ColsAtCompileTime == 1)
                std::cerr << "The object is not a column vector" << std::endl;
              else
                std::cerr << "The object is not a row vector" << std::endl;
#endif
              return 0;
            }
            
            if(size_at_compile_time != Eigen::Dynamic)
            { // This is a fixe size vector
              const Eigen::DenseIndex pyArray_size
              = PyArray_DIMS(pyArray)[0] > PyArray_DIMS(pyArray)[1]
              ? PyArray_DIMS(pyArray)[0]
              : PyArray_DIMS(pyArray)[1];
              if(size_at_compile_time != pyArray_size)
                return 0;
            }
            break;
          }
          default:
            return 0;
        }
      }
      else // this is a matrix
      {
        if(PyArray_NDIM(pyArray) == 1) // We can always convert a vector into a matrix
        {
          return pyArray;
        }
        
        if(PyArray_NDIM(pyArray) != 2)
        {
#ifndef NDEBUG
            std::cerr << "The number of dimension of the object is not correct." << std::endl;
#endif
          return 0;
        }
       
        if(PyArray_NDIM(pyArray) == 2)
        {
          const int R = (int)PyArray_DIMS(pyArray)[0];
          const int C = (int)PyArray_DIMS(pyArray)[1];
          
          if( (MatType::RowsAtCompileTime!=R)
             && (MatType::RowsAtCompileTime!=Eigen::Dynamic) )
            return 0;
          if( (MatType::ColsAtCompileTime!=C)
             && (MatType::ColsAtCompileTime!=Eigen::Dynamic) )
            return 0;
        }
      }
        
#ifdef NPY_1_8_API_VERSION
      if(!(PyArray_FLAGS(pyArray)))
#else
      if(!(PyArray_FLAGS(pyArray) & NPY_ALIGNED))
#endif
      {
#ifndef NDEBUG
        std::cerr << "NPY non-aligned matrices are not implemented." << std::endl;
#endif
        return 0;
      }
      
      return pyArray;
    }
 
    /// \brief Allocate memory and copy pyObj in the new storage
    static void construct(PyObject* pyObj,
                          bp::converter::rvalue_from_python_stage1_data* memory)
    {
      PyArrayObject * pyArray = reinterpret_cast<PyArrayObject*>(pyObj);
      assert((PyArray_DIMS(pyArray)[0]<INT_MAX) && (PyArray_DIMS(pyArray)[1]<INT_MAX));
      
      void* storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<MatType>*>
                     (reinterpret_cast<void*>(memory))->storage.bytes;
      
      EigenObjectAllocator<MatType>::allocate(pyArray,storage);

      memory->convertible = storage;
    }
    
    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
       &EigenFromPy::construct,bp::type_id<MatType>());
    }
  };
  
  template<typename MatType>
  struct EigenFromPyConverter
  {
    static void registration()
    {
      EigenFromPy<MatType>::registration();

      // Add also conversion to Eigen::MatrixBase<MatType>
      typedef Eigen::MatrixBase<MatType> MatrixBase;
      EigenFromPy<MatrixBase>::registration();

      // Add also conversion to Eigen::EigenBase<MatType>
      typedef Eigen::EigenBase<MatType> EigenBase;
      EigenFromPy<EigenBase>::registration();
    }
  };

  template<typename MatType>
  struct EigenFromPy< Eigen::MatrixBase<MatType> > : EigenFromPy<MatType>
  {
    typedef EigenFromPy<MatType> EigenFromPyDerived;
    typedef Eigen::MatrixBase<MatType> Base;

    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
       &EigenFromPy::construct,bp::type_id<Base>());
    }
  };
    
  template<typename MatType>
  struct EigenFromPy< Eigen::EigenBase<MatType> > : EigenFromPy<MatType>
  {
    typedef EigenFromPy<MatType> EigenFromPyDerived;
    typedef Eigen::EigenBase<MatType> Base;

    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
       &EigenFromPy::construct,bp::type_id<Base>());
    }
  };

#if EIGEN_VERSION_AT_LEAST(3,2,0)
  // Template specialization for Eigen::Ref
  template<typename MatType>
  struct EigenFromPyConverter< eigenpy::Ref<MatType> >
  {
    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy<MatType>::convertible),
       &EigenFromPy<MatType>::construct,bp::type_id<MatType>());
    }
  };
#endif

  
  template<typename MatType,typename EigenEquivalentType>
  EIGENPY_DEPRECATED
  void enableEigenPySpecific()
  {
    enableEigenPySpecific<MatType>();
  }

  template<typename MatType>
  void enableEigenPySpecific()
  {
    if(check_registration<MatType>()) return;
    
    bp::to_python_converter<MatType,EigenToPy<MatType> >();
    EigenFromPyConverter<MatType>::registration();
  }

} // namespace eigenpy

#endif // ifndef __eigenpy_details_hpp__
