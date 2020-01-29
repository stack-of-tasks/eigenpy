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

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/registration.hpp"
#include "eigenpy/map.hpp"
#include "eigenpy/exception.hpp"

namespace boost { namespace python { namespace detail {

  template<class MatType>
  struct referent_size<Eigen::MatrixBase<MatType>&>
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

}}}

namespace boost { namespace python { namespace converter {

template<class MatType>
struct implicit<Eigen::MatrixBase<MatType>,MatType>
{
  typedef Eigen::MatrixBase<MatType> Source;
  typedef MatType Target;
  
  static void* convertible(PyObject* obj)
  {
    // Find a converter which can produce a Source instance from
    // obj. The user has told us that Source can be converted to
    // Target, and instantiating construct() below, ensures that
    // at compile-time.
    return implicit_rvalue_convertible_from_python(obj, registered<Source>::converters)
    ? obj : 0;
  }
  
  static void construct(PyObject* obj, rvalue_from_python_stage1_data* data)
  {
    void* storage = ((rvalue_from_python_storage<Target>*)data)->storage.bytes;
    
    arg_from_python<Source> get_source(obj);
    bool convertible = get_source.convertible();
    BOOST_VERIFY(convertible);
    
    new (storage) Target(get_source().derived());
    
    // record successful construction
    data->convertible = storage;
  }
};

template<class MatType>
struct implicit<MatType,Eigen::MatrixBase<MatType> >
{
  typedef MatType Source;
  typedef Eigen::MatrixBase<MatType> Target;
  
  static void* convertible(PyObject* obj)
  {
    // Find a converter which can produce a Source instance from
    // obj. The user has told us that Source can be converted to
    // Target, and instantiating construct() below, ensures that
    // at compile-time.
    return implicit_rvalue_convertible_from_python(obj, registered<Source>::converters)
    ? obj : 0;
  }
  
  static void construct(PyObject* obj, rvalue_from_python_stage1_data* data)
  {
    void* storage = reinterpret_cast<rvalue_from_python_storage<Target>*>
                   (reinterpret_cast<void*>(data))->storage.bytes;
    
    arg_from_python<Source> get_source(obj);
    bool convertible = get_source.convertible();
    BOOST_VERIFY(convertible);
    
    new (storage) Source(get_source());
    
    // record successful construction
    data->convertible = storage;
  }
};

template<class MatType>
struct implicit<MatType,Eigen::EigenBase<MatType> > : implicit<MatType,Eigen::MatrixBase<MatType> >
{};

}}} // namespace boost::python::converter

#define GET_PY_ARRAY_TYPE(array) PyArray_ObjectType(reinterpret_cast<PyObject *>(array), 0)

namespace eigenpy
{
  template <typename SCALAR>  struct NumpyEquivalentType {};
  template <> struct NumpyEquivalentType<float>   { enum { type_code = NPY_FLOAT  };};
  template <> struct NumpyEquivalentType< std::complex<float> >   { enum { type_code = NPY_CFLOAT  };};
  template <> struct NumpyEquivalentType<double>  { enum { type_code = NPY_DOUBLE };};
  template <> struct NumpyEquivalentType< std::complex<double> >  { enum { type_code = NPY_CDOUBLE };};
  template <> struct NumpyEquivalentType<long double>  { enum { type_code = NPY_LONGDOUBLE };};
  template <> struct NumpyEquivalentType< std::complex<long double> >  { enum { type_code = NPY_CLONGDOUBLE };};
  template <> struct NumpyEquivalentType<int>     { enum { type_code = NPY_INT    };};
  template <> struct NumpyEquivalentType<long>    { enum { type_code = NPY_LONG    };};

  template <typename SCALAR1, typename SCALAR2>
  struct FromTypeToType : public boost::false_type {};
  
  template <typename SCALAR>
  struct FromTypeToType<SCALAR,SCALAR> : public boost::true_type {};
  
  template <> struct FromTypeToType<int,long> : public boost::true_type {};
  template <> struct FromTypeToType<int,float> : public boost::true_type {};
  template <> struct FromTypeToType<int,std::complex<float> > : public boost::true_type {};
  template <> struct FromTypeToType<int,double> : public boost::true_type {};
  template <> struct FromTypeToType<int,std::complex<double> > : public boost::true_type {};
  template <> struct FromTypeToType<int,long double> : public boost::true_type {};
  template <> struct FromTypeToType<int,std::complex<long double> > : public boost::true_type {};
  
  template <> struct FromTypeToType<long,float> : public boost::true_type {};
  template <> struct FromTypeToType<long,std::complex<float> > : public boost::true_type {};
  template <> struct FromTypeToType<long,double> : public boost::true_type {};
  template <> struct FromTypeToType<long,std::complex<double> > : public boost::true_type {};
  template <> struct FromTypeToType<long,long double> : public boost::true_type {};
  template <> struct FromTypeToType<long,std::complex<long double> > : public boost::true_type {};
  
  template <> struct FromTypeToType<float,std::complex<float> > : public boost::true_type {};
  template <> struct FromTypeToType<float,double> : public boost::true_type {};
  template <> struct FromTypeToType<float,std::complex<double> > : public boost::true_type {};
  template <> struct FromTypeToType<float,long double> : public boost::true_type {};
  template <> struct FromTypeToType<float,std::complex<long double> > : public boost::true_type {};

  template <> struct FromTypeToType<double,std::complex<double> > : public boost::true_type {};
  template <> struct FromTypeToType<double,long double> : public boost::true_type {};
  template <> struct FromTypeToType<double,std::complex<long double> > : public boost::true_type {};

  namespace bp = boost::python;

  enum NP_TYPE
  {
    DEFAULT_TYPE,
    MATRIX_TYPE,
    ARRAY_TYPE
  };
  
  struct NumpyType
  {
    
    static NumpyType & getInstance()
    {
      static NumpyType instance;
      return instance;
    }

    operator bp::object () { return CurrentNumpyType; }

    bp::object make(PyArrayObject* pyArray, bool copy = false)
    { return make((PyObject*)pyArray,copy); }
    
    bp::object make(PyObject* pyObj, bool copy = false)
    {
      if (getType() == DEFAULT_TYPE) {
        std::cerr <<
          "eigenpy warning: you use the deprecated class numpy.matrix without explicily asking for it. "
          "The default behaviour will change to numpy.array at next major release.\n"
          "- Either call eigenpy.switchToNumpyMatrix() before using eigenpy to suppress this warning\n"
          "- or call eigenpy.switchToNumpyArray() and adapt your code accordingly.\n"
          "See https://github.com/stack-of-tasks/eigenpy/issues/87 for further details."
          << std::endl;
        switchToNumpyMatrix();
      }
      bp::object m;
      if(PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(CurrentNumpyType.ptr()),NumpyMatrixType))
        m = NumpyMatrixObject(bp::object(bp::handle<>(pyObj)), bp::object(), copy);
//        m = NumpyAsMatrixObject(bp::object(bp::handle<>(pyObj)));
      else if(PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(CurrentNumpyType.ptr()),NumpyArrayType))
        m = bp::object(bp::handle<>(pyObj)); // nothing to do here

      Py_INCREF(m.ptr());
      return m;
    }
    
    static void setNumpyType(bp::object & obj)
    {
      PyTypeObject * obj_type = PyType_Check(obj.ptr()) ? reinterpret_cast<PyTypeObject*>(obj.ptr()) : obj.ptr()->ob_type;
      if(PyType_IsSubtype(obj_type,getInstance().NumpyMatrixType))
        switchToNumpyMatrix();
      else if(PyType_IsSubtype(obj_type,getInstance().NumpyArrayType))
        switchToNumpyArray();
    }
    
    static void switchToNumpyArray()
    {
      getInstance().CurrentNumpyType = getInstance().NumpyArrayObject;
      getType() = ARRAY_TYPE;
    }
    
    static void switchToNumpyMatrix()
    {
      getInstance().CurrentNumpyType = getInstance().NumpyMatrixObject;
      getType() = MATRIX_TYPE;
    }
    
    static NP_TYPE & getType()
    {
      static NP_TYPE np_type;
      return np_type;
    }
    
    static bp::object getNumpyType()
    {
      return getInstance().CurrentNumpyType;
    }
    
    static const PyTypeObject * getNumpyMatrixType()
    {
      return getInstance().NumpyMatrixType;
    }
    
    static const PyTypeObject * getNumpyArrayType()
    {
      return getInstance().NumpyArrayType;
    }
    
    static bool isMatrix()
    {
      return PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(getInstance().CurrentNumpyType.ptr()),
                              getInstance().NumpyMatrixType);
    }
    
    static bool isArray()
    {
      return PyType_IsSubtype(reinterpret_cast<PyTypeObject*>(getInstance().CurrentNumpyType.ptr()),
                              getInstance().NumpyArrayType);
    }

  protected:
    NumpyType()
    {
      pyModule = bp::import("numpy");
#if PY_MAJOR_VERSION >= 3
      // TODO I don't know why this Py_INCREF is necessary.
      // Without it, the destructor of NumpyType SEGV sometimes.
      Py_INCREF(pyModule.ptr());
#endif
      
      NumpyMatrixObject = pyModule.attr("matrix");
      NumpyMatrixType = reinterpret_cast<PyTypeObject*>(NumpyMatrixObject.ptr());
      NumpyArrayObject = pyModule.attr("ndarray");
      NumpyArrayType = reinterpret_cast<PyTypeObject*>(NumpyArrayObject.ptr());
      //NumpyAsMatrixObject = pyModule.attr("asmatrix");
      //NumpyAsMatrixType = reinterpret_cast<PyTypeObject*>(NumpyAsMatrixObject.ptr());
      
      CurrentNumpyType = NumpyMatrixObject; // default conversion
      getType() = DEFAULT_TYPE;
    }

    bp::object CurrentNumpyType;
    bp::object pyModule;
    
    // Numpy types
    bp::object NumpyMatrixObject; PyTypeObject * NumpyMatrixType;
    //bp::object NumpyAsMatrixObject; PyTypeObject * NumpyAsMatrixType;
    bp::object NumpyArrayObject; PyTypeObject * NumpyArrayType;
    
  };

  template<typename MatType, bool IsVectorAtCompileTime = MatType::IsVectorAtCompileTime>
  struct initEigenObject
  {
    static MatType * run(PyArrayObject * pyArray, void * storage)
    {
      assert(PyArray_NDIM(pyArray) == 2);

      const int rows = (int)PyArray_DIMS(pyArray)[0];
      const int cols = (int)PyArray_DIMS(pyArray)[1];
      
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
    static void run(const Eigen::MatrixBase<MatrixIn> & input, const Eigen::MatrixBase<MatrixOut> & dest)
    {
      MatrixOut & dest_ = const_cast<MatrixOut &>(dest.derived());
      dest_ = input.template cast<NewScalar>();
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
      
      const int pyArray_Type = GET_PY_ARRAY_TYPE(pyArray);
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
      const int pyArray_Type = GET_PY_ARRAY_TYPE(pyArray);
      
      if(pyArray_Type == NumpyEquivalentType<Scalar>::type_code)
      {
        MapNumpy<MatType,Scalar>::map(pyArray) = mat; // no cast needed
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
  
  /* --- TO PYTHON -------------------------------------------------------------- */
  
  template<typename MatType>
  struct EigenToPy
  {
    static PyObject* convert(MatType const & mat)
    {
      typedef typename MatType::Scalar Scalar;
      assert( (mat.rows()<INT_MAX) && (mat.cols()<INT_MAX) 
	      && "Matrix range larger than int ... should never happen." );
      const int R  = (int)mat.rows(), C = (int)mat.cols();

      PyArrayObject* pyArray;
      // Allocate Python memory
      if(C == 1 && NumpyType::getType() == ARRAY_TYPE && MatType::IsVectorAtCompileTime) // Handle array with a single dimension
      {
        npy_intp shape[1] = { R };
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
        if(PyArray_NDIM(pyArray) != 2)
        {
          if ( (PyArray_NDIM(pyArray) !=1) || (! MatType::IsVectorAtCompileTime) )
          {
#ifndef NDEBUG
            std::cerr << "The number of dimension of the object is not correct." << std::endl;
#endif
            return 0;
          }
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
  
  template<typename MatType,typename EigenEquivalentType>
  EIGENPY_DEPRECATED
  void enableEigenPySpecific()
  {
    enableEigenPySpecific<MatType>();
  }
  
  template<typename MatType>
  struct EigenFromPyConverter
  {
    static void registration()
    {
      EigenFromPy<MatType>::registration();

      // Add also conversion to Eigen::MatrixBase<MatType>
      typedef Eigen::MatrixBase<MatType> MatrixBase;
//      bp::implicitly_convertible<MatTypeBase,MatType>();
      bp::implicitly_convertible<MatType,MatrixBase>();
      
      // Add also conversion to Eigen::EigenBase<MatType>
      typedef Eigen::EigenBase<MatType> EigenBase;
      bp::implicitly_convertible<MatType,EigenBase>();
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

#define numpy_import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }
  
  template<typename MatType>
  void enableEigenPySpecific()
  {
    numpy_import_array();
    if(check_registration<MatType>()) return;
    
    bp::to_python_converter<MatType,EigenToPy<MatType> >();
    EigenFromPyConverter<MatType>::registration();
  }

} // namespace eigenpy

#endif // ifndef __eigenpy_details_hpp__
