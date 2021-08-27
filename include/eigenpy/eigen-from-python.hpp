//
// Copyright (c) 2014-2020 CNRS INRIA
//

#ifndef __eigenpy_eigen_from_python_hpp__
#define __eigenpy_eigen_from_python_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/eigen-allocator.hpp"
#include "eigenpy/scalar-conversion.hpp"

#include <boost/python/converter/rvalue_from_python_data.hpp>

namespace eigenpy
{
  namespace details
  {
    template<typename MatType, bool is_const = boost::is_const<MatType>::value>
    struct copy_if_non_const
    {
      static void run(const Eigen::MatrixBase<MatType> & input,
                      PyArrayObject * pyArray)
      {
        EigenAllocator<MatType>::copy(input,pyArray);
      }
    };
  
    template<typename MatType>
    struct copy_if_non_const<const MatType,true>
    {
      static void run(const Eigen::MatrixBase<MatType> & /*input*/,
                      PyArrayObject * /*pyArray*/)
      {}
    };
  
#if EIGEN_VERSION_AT_LEAST(3,2,0)
    template<typename MatType, int Options, typename Stride> struct referent_storage_eigen_ref;
  
    template<typename MatType, int Options, typename Stride>
    struct referent_storage_eigen_ref
    {
      typedef Eigen::Ref<MatType,Options,Stride> RefType;
#if BOOST_VERSION / 100 % 1000 >= 77
      typedef typename ::boost::python::detail::aligned_storage<
          ::boost::python::detail::referent_size<RefType&>::value,
          ::boost::alignment_of<RefType&>::value
      >::type AlignedStorage;
#else
      typedef ::boost::python::detail::aligned_storage<
          ::boost::python::detail::referent_size<RefType&>::value
      > AlignedStorage;
#endif
      
      referent_storage_eigen_ref()
      : pyArray(NULL)
      , mat_ptr(NULL)
      , ref_ptr(reinterpret_cast<RefType*>(ref_storage.bytes))
      {
      }
      
      referent_storage_eigen_ref(const RefType & ref,
                                 PyArrayObject * pyArray,
                                 MatType * mat_ptr = NULL)
      : pyArray(pyArray)
      , mat_ptr(mat_ptr)
      , ref_ptr(reinterpret_cast<RefType*>(ref_storage.bytes))
      {
        Py_INCREF(pyArray);
        new (ref_storage.bytes) RefType(ref);
      }
      
      ~referent_storage_eigen_ref()
      {
        if(mat_ptr != NULL && PyArray_ISWRITEABLE(pyArray))
          copy_if_non_const<MatType>::run(*mat_ptr,pyArray);
        
        Py_DECREF(pyArray);
          
        if(mat_ptr != NULL)
          mat_ptr->~MatType();
        
        ref_ptr->~RefType();
      }
 
      AlignedStorage ref_storage;
      PyArrayObject * pyArray;
      MatType * mat_ptr;
      RefType * ref_ptr;
    };
#endif
    
  }
}

namespace boost { namespace python { namespace detail {
#if EIGEN_VERSION_AT_LEAST(3,2,0)
  template<typename MatType, int Options, typename Stride>
  struct referent_storage<Eigen::Ref<MatType,Options,Stride> &>
  {
    typedef ::eigenpy::details::referent_storage_eigen_ref<MatType,Options,Stride> StorageType;
#if BOOST_VERSION / 100 % 1000 >= 77
    typedef typename aligned_storage<referent_size<StorageType&>::value>::type type;
#else
    typedef aligned_storage<referent_size<StorageType&>::value> type;
#endif
  };

  template<typename MatType, int Options, typename Stride>
  struct referent_storage<const Eigen::Ref<const MatType,Options,Stride> &>
  {
    typedef ::eigenpy::details::referent_storage_eigen_ref<const MatType,Options,Stride> StorageType;
#if BOOST_VERSION / 100 % 1000 >= 77
    typedef typename aligned_storage<referent_size<StorageType&>::value, alignment_of<StorageType&>::value>::type type;
#else
    typedef aligned_storage<referent_size<StorageType&>::value> type;
#endif
  };
#endif
}}}

namespace boost { namespace python { namespace converter {

  template<typename MatrixReference>
  struct rvalue_from_python_data_eigen
  : rvalue_from_python_storage<MatrixReference>
  {
    typedef MatrixReference T;
    
# if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) \
&& (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) \
&& (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) \
&& !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing this */
    // This must always be a POD struct with m_data its first member.
    BOOST_STATIC_ASSERT(BOOST_PYTHON_OFFSETOF(rvalue_from_python_storage<T>,stage1) == 0);
# endif
    
    // The usual constructor
    rvalue_from_python_data_eigen(rvalue_from_python_stage1_data const & _stage1)
    {
      this->stage1 = _stage1;
    }
    
    // This constructor just sets m_convertible -- used by
    // implicitly_convertible<> to perform the final step of the
    // conversion, where the construct() function is already known.
    rvalue_from_python_data_eigen(void* convertible)
    {
      this->stage1.convertible = convertible;
    }
    
    // Destroys any object constructed in the storage.
    ~rvalue_from_python_data_eigen()
    {
      typedef typename boost::remove_const<typename boost::remove_reference<MatrixReference>::type>::type MatrixType;
      if (this->stage1.convertible == this->storage.bytes)
        static_cast<MatrixType *>((void *)this->storage.bytes)->~MatrixType();
    }
  };

#define EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(type)                                 \
  typedef rvalue_from_python_data_eigen<type> Base;                        \
                                                                           \
  rvalue_from_python_data(rvalue_from_python_stage1_data const & _stage1)  \
  : Base(_stage1)                                                          \
  {}                                                                       \
                                                                           \
  rvalue_from_python_data(void* convertible) : Base(convertible) {};

  /// \brief Template specialization of rvalue_from_python_data
  template<typename Derived>
  struct rvalue_from_python_data<Eigen::MatrixBase<Derived> const &>
  : rvalue_from_python_data_eigen<Derived const &>
  {
    EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
  };

  /// \brief Template specialization of rvalue_from_python_data
  template<typename Derived>
  struct rvalue_from_python_data<Eigen::EigenBase<Derived> const &>
  : rvalue_from_python_data_eigen<Derived const &>
  {
    EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
  };

  /// \brief Template specialization of rvalue_from_python_data
  template<typename Derived>
  struct rvalue_from_python_data<Eigen::PlainObjectBase<Derived> const &>
  : rvalue_from_python_data_eigen<Derived const &>
  {
    EIGENPY_RVALUE_FROM_PYTHON_DATA_INIT(Derived const &)
  };

  template<typename MatType, int Options, typename Stride>
  struct rvalue_from_python_data<Eigen::Ref<MatType,Options,Stride> &>
  : rvalue_from_python_storage<Eigen::Ref<MatType,Options,Stride> &>
  {
    typedef Eigen::Ref<MatType,Options,Stride> T;

# if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) \
&& (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) \
&& (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) \
&& !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing this */
    // This must always be a POD struct with m_data its first member.
    BOOST_STATIC_ASSERT(BOOST_PYTHON_OFFSETOF(rvalue_from_python_storage<T>,stage1) == 0);
# endif

    // The usual constructor
    rvalue_from_python_data(rvalue_from_python_stage1_data const & _stage1)
    {
      this->stage1 = _stage1;
    }

    // This constructor just sets m_convertible -- used by
    // implicitly_convertible<> to perform the final step of the
    // conversion, where the construct() function is already known.
    rvalue_from_python_data(void* convertible)
    {
      this->stage1.convertible = convertible;
    }

    // Destroys any object constructed in the storage.
    ~rvalue_from_python_data()
    {
      typedef ::eigenpy::details::referent_storage_eigen_ref<MatType, Options,Stride> StorageType;
      if (this->stage1.convertible == this->storage.bytes)
        static_cast<StorageType *>((void *)this->storage.bytes)->~StorageType();
    }
  };

  template<typename MatType, int Options, typename Stride>
  struct rvalue_from_python_data<const Eigen::Ref<const MatType,Options,Stride> &>
  : rvalue_from_python_storage<const Eigen::Ref<const MatType,Options,Stride> &>
  {
    typedef const Eigen::Ref<const MatType,Options,Stride> T;

# if (!defined(__MWERKS__) || __MWERKS__ >= 0x3000) \
&& (!defined(__EDG_VERSION__) || __EDG_VERSION__ >= 245) \
&& (!defined(__DECCXX_VER) || __DECCXX_VER > 60590014) \
&& !defined(BOOST_PYTHON_SYNOPSIS) /* Synopsis' OpenCXX has trouble parsing this */
    // This must always be a POD struct with m_data its first member.
    BOOST_STATIC_ASSERT(BOOST_PYTHON_OFFSETOF(rvalue_from_python_storage<T>,stage1) == 0);
# endif

    // The usual constructor
    rvalue_from_python_data(rvalue_from_python_stage1_data const & _stage1)
    {
      this->stage1 = _stage1;
    }

    // This constructor just sets m_convertible -- used by
    // implicitly_convertible<> to perform the final step of the
    // conversion, where the construct() function is already known.
    rvalue_from_python_data(void* convertible)
    {
      this->stage1.convertible = convertible;
    }

    // Destroys any object constructed in the storage.
    ~rvalue_from_python_data()
    {
      typedef ::eigenpy::details::referent_storage_eigen_ref<const MatType, Options,Stride> StorageType;
      if (this->stage1.convertible == this->storage.bytes)
        static_cast<StorageType *>((void *)this->storage.bytes)->~StorageType();
    }
  };

} } }

namespace eigenpy
{

  template<typename MatOrRefType>
  void eigen_from_py_construct(PyObject* pyObj,
                               bp::converter::rvalue_from_python_stage1_data* memory)
  {
    PyArrayObject * pyArray = reinterpret_cast<PyArrayObject*>(pyObj);
    assert((PyArray_DIMS(pyArray)[0]<INT_MAX) && (PyArray_DIMS(pyArray)[1]<INT_MAX));
    
    bp::converter::rvalue_from_python_storage<MatOrRefType>* storage = reinterpret_cast<bp::converter::rvalue_from_python_storage<MatOrRefType>*>
    (reinterpret_cast<void*>(memory));
    
    EigenAllocator<MatOrRefType>::allocate(pyArray,storage);

    memory->convertible = storage->storage.bytes;
  }

  template<typename MatType, typename _Scalar>
  struct EigenFromPy
  {
    typedef typename MatType::Scalar Scalar;
    
    /// \brief Determine if pyObj can be converted into a MatType object
    static void* convertible(PyObject* pyObj);
 
    /// \brief Allocate memory and copy pyObj in the new storage
    static void construct(PyObject* pyObj,
                          bp::converter::rvalue_from_python_stage1_data* memory);
    
    static void registration();
  };

  template<typename MatType, typename _Scalar>
  void* EigenFromPy<MatType,_Scalar>::convertible(PyObject* pyObj)
  {
    if(!call_PyArray_Check(reinterpret_cast<PyObject*>(pyObj)))
      return 0;
    
    PyArrayObject * pyArray =  reinterpret_cast<PyArrayObject*>(pyObj);
    
    if(!np_type_is_convertible_into_scalar<Scalar>(EIGENPY_GET_PY_ARRAY_TYPE(pyArray)))
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
            return 0;
          }
          
          if(((PyArray_DIMS(pyArray)[0] == 1) && (MatType::ColsAtCompileTime == 1))
             || ((PyArray_DIMS(pyArray)[1] == 1) && (MatType::RowsAtCompileTime == 1)))
          {
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
      return 0;
    }
    
    return pyArray;
  }

  template<typename MatType, typename _Scalar>
  void EigenFromPy<MatType,_Scalar>::construct(PyObject* pyObj,
                                               bp::converter::rvalue_from_python_stage1_data* memory)
  {
    eigen_from_py_construct<MatType>(pyObj,memory);
  }

  template<typename MatType, typename _Scalar>
  void EigenFromPy<MatType,_Scalar>::registration()
  {
    bp::converter::registry::push_back
    (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
     &EigenFromPy::construct,bp::type_id<MatType>());
  }
  
  template<typename MatType>
  struct EigenFromPyConverter
  {
    static void registration()
    {
      EigenFromPy<MatType>::registration();

      // Add conversion to Eigen::MatrixBase<MatType>
      typedef Eigen::MatrixBase<MatType> MatrixBase;
      EigenFromPy<MatrixBase>::registration();

      // Add conversion to Eigen::EigenBase<MatType>
      typedef Eigen::EigenBase<MatType> EigenBase;
      EigenFromPy<EigenBase,typename MatType::Scalar>::registration();

      // Add conversion to Eigen::PlainObjectBase<MatType>
      typedef Eigen::PlainObjectBase<MatType> PlainObjectBase;
      EigenFromPy<PlainObjectBase>::registration();

#if EIGEN_VERSION_AT_LEAST(3,2,0)
      // Add conversion to Eigen::Ref<MatType>
      typedef Eigen::Ref<MatType> RefType;
      EigenFromPy<RefType>::registration();
      
      // Add conversion to Eigen::Ref<MatType>
      typedef const Eigen::Ref<const MatType> ConstRefType;
      EigenFromPy<ConstRefType>::registration();
#endif
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
  struct EigenFromPy< Eigen::EigenBase<MatType>, typename MatType::Scalar > : EigenFromPy<MatType>
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
    
  template<typename MatType>
  struct EigenFromPy< Eigen::PlainObjectBase<MatType> > : EigenFromPy<MatType>
  {
    typedef EigenFromPy<MatType> EigenFromPyDerived;
    typedef Eigen::PlainObjectBase<MatType> Base;

    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
       &EigenFromPy::construct,bp::type_id<Base>());
    }
  };

#if EIGEN_VERSION_AT_LEAST(3,2,0)

  template<typename MatType, int Options, typename Stride>
  struct EigenFromPy<Eigen::Ref<MatType,Options,Stride> >
  {
    typedef Eigen::Ref<MatType,Options,Stride> RefType;
    typedef typename MatType::Scalar Scalar;
    
    /// \brief Determine if pyObj can be converted into a MatType object
    static void* convertible(PyObject * pyObj)
    {
      if(!call_PyArray_Check(pyObj))
        return 0;
      PyArrayObject * pyArray = reinterpret_cast<PyArrayObject*>(pyObj);
      if(!PyArray_ISWRITEABLE(pyArray))
        return 0;
      return EigenFromPy<MatType>::convertible(pyObj);
    }
    
    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
       &eigen_from_py_construct<RefType>,bp::type_id<RefType>());
    }
  };

  template<typename MatType, int Options, typename Stride>
  struct EigenFromPy<const Eigen::Ref<const MatType,Options,Stride> >
  {
    typedef const Eigen::Ref<const MatType,Options,Stride> ConstRefType;
    typedef typename MatType::Scalar Scalar;
    
    /// \brief Determine if pyObj can be converted into a MatType object
    static void* convertible(PyObject * pyObj)
    {
      return EigenFromPy<MatType>::convertible(pyObj);
    }
    
    static void registration()
    {
      bp::converter::registry::push_back
      (reinterpret_cast<void *(*)(_object *)>(&EigenFromPy::convertible),
       &eigen_from_py_construct<ConstRefType>,bp::type_id<ConstRefType>());
    }
  };
#endif

}

#endif // __eigenpy_eigen_from_python_hpp__
