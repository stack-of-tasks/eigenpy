//
// Copyright (c) 2014-2020 CNRS INRIA
//

#ifndef __eigenpy_eigen_allocator_hpp__
#define __eigenpy_eigen_allocator_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-map.hpp"
#include "eigenpy/register.hpp"
#include "eigenpy/scalar-conversion.hpp"
#include "eigenpy/utils/is-aligned.hpp"

namespace eigenpy
{

  namespace details
  {
    template<typename MatType, bool IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
    struct init_matrix_or_array
    {
      static MatType * run(int rows, int cols, void * storage)
      {
        if(storage)
          return new (storage) MatType(rows,cols);
        else
          return new MatType(rows,cols);
      }
      
      static MatType * run(PyArrayObject * pyArray, void * storage = NULL)
      {
        assert(PyArray_NDIM(pyArray) == 1 || PyArray_NDIM(pyArray) == 2);

        int rows = -1, cols = -1;
        const int ndim = PyArray_NDIM(pyArray);
        if(ndim == 2)
        {
          rows = (int)PyArray_DIMS(pyArray)[0];
          cols = (int)PyArray_DIMS(pyArray)[1];
        }
        else if(ndim == 1)
        {
          rows = (int)PyArray_DIMS(pyArray)[0];
          cols = 1;
        }
  
        return run(rows,cols,storage);
      }
    };

    template<typename MatType>
    struct init_matrix_or_array<MatType,true>
    {
      static MatType * run(int rows, int cols, void * storage)
      {
        if(storage)
          return new (storage) MatType(rows,cols);
        else
          return new MatType(rows,cols);
      }
      
      static MatType * run(int size, void * storage)
      {
        if(storage)
          return new (storage) MatType(size);
        else
          return new MatType(size);
      }
      
      static MatType * run(PyArrayObject * pyArray, void * storage = NULL)
      {
        const int ndim = PyArray_NDIM(pyArray);
        if(ndim == 1)
        {
          const int size = (int)PyArray_DIMS(pyArray)[0];
          return run(size,storage);
        }
        else
        {
          const int rows = (int)PyArray_DIMS(pyArray)[0];
          const int cols = (int)PyArray_DIMS(pyArray)[1];
          return run(rows,cols,storage);
        }
      }
    };
  
    template<typename MatType>
    bool check_swap(PyArrayObject * pyArray,
                    const Eigen::MatrixBase<MatType> & mat)
    {
      if(PyArray_NDIM(pyArray) == 0) return false;
      if(mat.rows() == PyArray_DIMS(pyArray)[0])
        return false;
      else
        return true;
    }

    template<typename Scalar, typename NewScalar, bool cast_is_valid = FromTypeToType<Scalar,NewScalar>::value >
    struct cast_matrix_or_array
    {
      template<typename MatrixIn, typename MatrixOut>
      static void run(const Eigen::MatrixBase<MatrixIn> & input,
                      const Eigen::MatrixBase<MatrixOut> & dest)
      {
        MatrixOut & dest_ = const_cast<MatrixOut &>(dest.derived());
        dest_ = input.template cast<NewScalar>();
      }
    };

    template<typename Scalar, typename NewScalar>
    struct cast_matrix_or_array<Scalar,NewScalar,false>
    {
      template<typename MatrixIn, typename MatrixOut>
      static void run(const Eigen::MatrixBase<MatrixIn> & /*input*/,
                      const Eigen::MatrixBase<MatrixOut> & /*dest*/)
      {
        // do nothing
        assert(false && "Must never happened");
      }
    };
  
  } // namespace details

#define EIGENPY_CAST_FROM_PYARRAY_TO_EIGEN_MATRIX(MatType,Scalar,NewScalar,pyArray,mat) \
  details::cast_matrix_or_array<Scalar,NewScalar>::run(NumpyMap<MatType,Scalar>::map(pyArray,details::check_swap(pyArray,mat)),mat)

#define EIGENPY_CAST_FROM_EIGEN_MATRIX_TO_PYARRAY(MatType,Scalar,NewScalar,mat,pyArray) \
  details::cast_matrix_or_array<Scalar,NewScalar>::run(mat,NumpyMap<MatType,NewScalar>::map(pyArray,details::check_swap(pyArray,mat)))

  template<typename MatType>
  struct EigenAllocator
  {
    typedef MatType Type;
    typedef typename MatType::Scalar Scalar;
    
    static void allocate(PyArrayObject * pyArray,
                         bp::converter::rvalue_from_python_storage<MatType> * storage)
    {
      void * raw_ptr = storage->storage.bytes;
      Type * mat_ptr = details::init_matrix_or_array<Type>::run(pyArray,raw_ptr);
      Type & mat = *mat_ptr;
      
      const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      if(pyArray_type_code == Scalar_type_code)
      {
        mat = NumpyMap<MatType,Scalar>::map(pyArray,details::check_swap(pyArray,mat)); // avoid useless cast
        return;
      }
      
      switch(pyArray_type_code)
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
      const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      
      typedef typename NumpyMap<MatType,Scalar>::EigenMap MapType;
      
      if(pyArray_type_code == Scalar_type_code) // no cast needed
      {
        MapType map_pyArray = NumpyMap<MatType,Scalar>::map(pyArray,details::check_swap(pyArray,mat));
        map_pyArray = mat;
        return;
      }
      
      switch(pyArray_type_code)
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
  template<typename MatType, int Options, typename Stride>
  struct EigenAllocator<Eigen::Ref<MatType,Options,Stride> >
  {
    typedef Eigen::Ref<MatType,Options,Stride> RefType;
    typedef typename MatType::Scalar Scalar;

    typedef typename ::boost::python::detail::referent_storage<RefType&>::StorageType StorageType;
    
    static void allocate(PyArrayObject * pyArray,
                         bp::converter::rvalue_from_python_storage<RefType> * storage)
    {
      typedef typename StrideType<MatType,Eigen::internal::traits<RefType>::StrideType::InnerStrideAtCompileTime, Eigen::internal::traits<RefType>::StrideType::OuterStrideAtCompileTime >::type NumpyMapStride;

      bool need_to_allocate = false;
      const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      if(pyArray_type_code != Scalar_type_code)
        need_to_allocate |= true;
      if(    (MatType::IsRowMajor && (PyArray_IS_C_CONTIGUOUS(pyArray) && !PyArray_IS_F_CONTIGUOUS(pyArray)))
          || (!MatType::IsRowMajor && (PyArray_IS_F_CONTIGUOUS(pyArray) && !PyArray_IS_C_CONTIGUOUS(pyArray)))
          || MatType::IsVectorAtCompileTime
          || (PyArray_IS_F_CONTIGUOUS(pyArray) && PyArray_IS_C_CONTIGUOUS(pyArray))) // no need to allocate
        need_to_allocate |= false;
      else
        need_to_allocate |= true;
      if(Options != Eigen::Unaligned) // we need to check whether the memory is correctly aligned and composed of a continuous segment
      {
        void * data_ptr = PyArray_DATA(pyArray);
        if(!PyArray_ISONESEGMENT(pyArray) || !is_aligned(data_ptr,Options))
          need_to_allocate |= true;
      }
      
      void * raw_ptr = storage->storage.bytes;
      if(need_to_allocate)
      {
        MatType * mat_ptr;
        mat_ptr = details::init_matrix_or_array<MatType>::run(pyArray);
        RefType mat_ref(*mat_ptr);
        
        new (raw_ptr) StorageType(mat_ref,pyArray,mat_ptr);
        
        RefType & mat = *reinterpret_cast<RefType*>(raw_ptr);
        if(pyArray_type_code == Scalar_type_code)
        {
          mat = NumpyMap<MatType,Scalar>::map(pyArray,details::check_swap(pyArray,mat)); // avoid useless cast
          return;
        }
        
        switch(pyArray_type_code)
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
      else
      {
        assert(pyArray_type_code == Scalar_type_code);
        typename NumpyMap<MatType,Scalar,Options,NumpyMapStride>::EigenMap numpyMap = NumpyMap<MatType,Scalar,Options,NumpyMapStride>::map(pyArray);
        RefType mat_ref(numpyMap);
        new (raw_ptr) StorageType(mat_ref,pyArray);
      }
    }
    
    static void copy(RefType const & ref, PyArrayObject * pyArray)
    {
      EigenAllocator<MatType>::copy(ref,pyArray);
    }
  };

  template<typename MatType, int Options, typename Stride>
  struct EigenAllocator<const Eigen::Ref<const MatType,Options,Stride> >
  {
    typedef const Eigen::Ref<const MatType,Options,Stride> RefType;
    typedef typename MatType::Scalar Scalar;

    typedef typename ::boost::python::detail::referent_storage<RefType&>::StorageType StorageType;
    
    static void allocate(PyArrayObject * pyArray,
                         bp::converter::rvalue_from_python_storage<RefType> * storage)
    {
      typedef typename StrideType<MatType,Eigen::internal::traits<RefType>::StrideType::InnerStrideAtCompileTime, Eigen::internal::traits<RefType>::StrideType::OuterStrideAtCompileTime >::type NumpyMapStride;

      bool need_to_allocate = false;
      const int pyArray_type_code = EIGENPY_GET_PY_ARRAY_TYPE(pyArray);
      const int Scalar_type_code = Register::getTypeCode<Scalar>();
      
      if(pyArray_type_code != Scalar_type_code)
        need_to_allocate |= true;
      if(    (MatType::IsRowMajor && (PyArray_IS_C_CONTIGUOUS(pyArray) && !PyArray_IS_F_CONTIGUOUS(pyArray)))
          || (!MatType::IsRowMajor && (PyArray_IS_F_CONTIGUOUS(pyArray) && !PyArray_IS_C_CONTIGUOUS(pyArray)))
          || MatType::IsVectorAtCompileTime
          || (PyArray_IS_F_CONTIGUOUS(pyArray) && PyArray_IS_C_CONTIGUOUS(pyArray))) // no need to allocate
        need_to_allocate |= false;
      else
        need_to_allocate |= true;
      if(Options != Eigen::Unaligned) // we need to check whether the memory is correctly aligned and composed of a continuous segment
      {
        void * data_ptr = PyArray_DATA(pyArray);
        if(!PyArray_ISONESEGMENT(pyArray) || !is_aligned(data_ptr,Options))
          need_to_allocate |= true;
      }
      
      void * raw_ptr = storage->storage.bytes;
      if(need_to_allocate)
      {
        MatType * mat_ptr;
        mat_ptr = details::init_matrix_or_array<MatType>::run(pyArray);
        RefType mat_ref(*mat_ptr);
        
        new (raw_ptr) StorageType(mat_ref,pyArray,mat_ptr);
        
        MatType & mat = *mat_ptr;
        if(pyArray_type_code == Scalar_type_code)
        {
          mat = NumpyMap<MatType,Scalar>::map(pyArray,details::check_swap(pyArray,mat)); // avoid useless cast
          return;
        }
        
        switch(pyArray_type_code)
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
      else
      {
        assert(pyArray_type_code == Scalar_type_code);
        typename NumpyMap<MatType,Scalar,Options,NumpyMapStride>::EigenMap numpyMap = NumpyMap<MatType,Scalar,Options,NumpyMapStride>::map(pyArray);
        RefType mat_ref(numpyMap);
        new (raw_ptr) StorageType(mat_ref,pyArray);
      }
    }
    
    static void copy(RefType const & ref, PyArrayObject * pyArray)
    {
      EigenAllocator<MatType>::copy(ref,pyArray);
    }
  };
#endif
}

#endif // __eigenpy_eigen_allocator_hpp__
