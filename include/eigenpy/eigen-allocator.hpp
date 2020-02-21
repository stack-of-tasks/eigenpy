//
// Copyright (c) 2014-2020 CNRS INRIA
//

#ifndef __eigenpy_eigen_allocator_hpp__
#define __eigenpy_eigen_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/map.hpp"
#include "eigenpy/scalar-conversion.hpp"

namespace eigenpy
{
  template<typename MatType, bool IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
  struct initEigenObject
  {
    static MatType * run(PyArrayObject * pyArray, void * storage)
    {
      assert(PyArray_NDIM(pyArray) == 1 || PyArray_NDIM(pyArray) == 2);

      int rows = -1, cols = -1;
      if(PyArray_NDIM(pyArray) == 2)
      {
        rows = (int)PyArray_DIMS(pyArray)[0];
        cols = (int)PyArray_DIMS(pyArray)[1];
      }
      else if(PyArray_NDIM(pyArray) == 1)
      {
        rows = (int)PyArray_DIMS(pyArray)[0];
        cols = 1;
      }
              
      return new (storage) MatType(rows,cols);
    }
  };

  template<typename MatType>
  struct initEigenObject<MatType,true>
  {
    static MatType * run(PyArrayObject * pyArray, void * storage)
    {
      if(PyArray_NDIM(pyArray) == 1)
      {
        const int rows_or_cols = (int)PyArray_DIMS(pyArray)[0];
        return new (storage) MatType(rows_or_cols);
      }
      else
      {
        const int rows = (int)PyArray_DIMS(pyArray)[0];
        const int cols = (int)PyArray_DIMS(pyArray)[1];
        return new (storage) MatType(rows,cols);
      }
    }
  };

  template<typename Scalar, typename NewScalar, bool cast_is_valid = FromTypeToType<Scalar,NewScalar>::value >
  struct CastMatToMat
  {
    template<typename MatrixIn, typename MatrixOut>
    static void run(const Eigen::MatrixBase<MatrixIn> & input,
                    const Eigen::MatrixBase<MatrixOut> & dest)
    {
      MatrixOut & dest_ = const_cast<MatrixOut &>(dest.derived());
      if(dest.rows() == input.rows())
        dest_ = input.template cast<NewScalar>();
      else
        dest_ = input.transpose().template cast<NewScalar>();
    }
  };

  template<typename Scalar, typename NewScalar>
  struct CastMatToMat<Scalar,NewScalar,false>
  {
    template<typename MatrixIn, typename MatrixOut>
    static void run(const Eigen::MatrixBase<MatrixIn> & /*input*/,
                    const Eigen::MatrixBase<MatrixOut> & /*dest*/)
    {
      // do nothing
      assert("Must never happened");
    }
  };

#define EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,Scalar,NewScalar,pyArray,mat) \
  CastMatToMat<Scalar,NewScalar>::run(MapNumpy<MatType,Scalar>::map(pyArray),mat)

#define EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,NewScalar,mat,pyArray) \
  CastMatToMat<Scalar,NewScalar>::run(mat,MapNumpy<MatType,NewScalar>::map(pyArray))
  
  template<typename MatType>
  struct EigenObjectAllocator
  {
    typedef MatType Type;
    typedef typename MatType::Scalar Scalar;
    
    static void allocate(PyArrayObject * pyArray, void * storage)
    {
      Type * mat_ptr = initEigenObject<Type>::run(pyArray,storage);
      Type & mat = *mat_ptr;
      
      const int pyArray_Type = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
      if(pyArray_Type == NumpyEquivalentType<Scalar>::type_code)
      {
        mat = MapNumpy<MatType,Scalar>::map(pyArray); // avoid useless cast
        return;
      }
      
      switch(pyArray_Type)
      {
        case NPY_INT:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,int,Scalar,pyArray,mat);
          break;
        case NPY_LONG:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,long,Scalar,pyArray,mat);
          break;
        case NPY_FLOAT:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,float,Scalar,pyArray,mat);
          break;
        case NPY_CFLOAT:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,std::complex<float>,Scalar,pyArray,mat);
          break;
        case NPY_DOUBLE:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,double,Scalar,pyArray,mat);
          break;
        case NPY_CDOUBLE:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,std::complex<double>,Scalar,pyArray,mat);
          break;
        case NPY_LONGDOUBLE:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,long double,Scalar,pyArray,mat);
          break;
        case NPY_CLONGDOUBLE:
          EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,std::complex<long double>,Scalar,pyArray,mat);
          break;
        default:
          throw Exception("You asked for a conversion which is not implemented.");
      }
    }
    
    /// \brief Copy mat into the Python array using Eigen::Map
    template<typename MatrixDerived>
    static void copy(const Eigen::MatrixBase<MatrixDerived> & mat_,
                     PyArrayObject * pyArray)
    {
      const MatrixDerived & mat = const_cast<const MatrixDerived &>(mat_.derived());
      const int pyArray_Type = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
      
      typedef typename MapNumpy<MatType,Scalar>::EigenMap MapType;
      
      if(pyArray_Type == NumpyEquivalentType<Scalar>::type_code) // no cast needed
      {
        MapType map_pyArray = MapNumpy<MatType,Scalar>::map(pyArray);
        if(mat.rows() == map_pyArray.rows())
          map_pyArray = mat;
        else
          map_pyArray = mat.transpose();
        return;
      }
      
      switch(pyArray_Type)
      {
        case NPY_INT:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,int,mat,pyArray);
          break;
        case NPY_LONG:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,long,mat,pyArray);
          break;
        case NPY_FLOAT:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,float,mat,pyArray);
          break;
        case NPY_CFLOAT:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,std::complex<float>,mat,pyArray);
          break;
        case NPY_DOUBLE:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,double,mat,pyArray);
          break;
        case NPY_CDOUBLE:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,std::complex<double>,mat,pyArray);
          break;
        case NPY_LONGDOUBLE:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,long double,mat,pyArray);
          break;
        case NPY_CLONGDOUBLE:
          EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,std::complex<long double>,mat,pyArray);
          break;
        default:
          throw Exception("You asked for a conversion which is not implemented.");
      }
    }
  };
  
#if EIGEN_VERSION_AT_LEAST(3,2,0)
  template<typename MatType>
  struct EigenObjectAllocator< eigenpy::Ref<MatType> >
  {
    typedef eigenpy::Ref<MatType> Type;
    typedef typename MatType::Scalar Scalar;
    
    static void allocate(PyArrayObject * pyArray, void * storage)
    {
      typename MapNumpy<MatType,Scalar>::EigenMap numpyMap = MapNumpy<MatType,Scalar>::map(pyArray);
      new (storage) Type(numpyMap);
    }
    
    static void copy(Type const & mat, PyArrayObject * pyArray)
    {
      EigenObjectAllocator<MatType>::copy(mat,pyArray);
    }
  };
#endif
}

#endif // __eigenpy_eigen_allocator_hpp__
