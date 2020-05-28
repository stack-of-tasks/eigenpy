/*
 * Copyright 2018-2020 INRIA
*/

#ifndef __eigenpy_numpy_type_hpp__
#define __eigenpy_numpy_type_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/scalar-conversion.hpp"

#include <stdexcept>
#include <typeinfo>
#include <sstream>

namespace eigenpy
{
  namespace bp = boost::python;

  // By default, the Scalar is considered as a Python object
  template <typename Scalar> struct NumpyEquivalentType { enum  { type_code = NPY_USERDEF };};

  template <> struct NumpyEquivalentType<float>   { enum { type_code = NPY_FLOAT  };};
  template <> struct NumpyEquivalentType< std::complex<float> >   { enum { type_code = NPY_CFLOAT  };};
  template <> struct NumpyEquivalentType<double>  { enum { type_code = NPY_DOUBLE };};
  template <> struct NumpyEquivalentType< std::complex<double> >  { enum { type_code = NPY_CDOUBLE };};
  template <> struct NumpyEquivalentType<long double>  { enum { type_code = NPY_LONGDOUBLE };};
  template <> struct NumpyEquivalentType< std::complex<long double> >  { enum { type_code = NPY_CLONGDOUBLE };};
  template <> struct NumpyEquivalentType<bool>    { enum { type_code = NPY_BOOL  };};
  template <> struct NumpyEquivalentType<int>     { enum { type_code = NPY_INT    };};
  template <> struct NumpyEquivalentType<long>    { enum { type_code = NPY_LONG    };};

  template<typename Scalar>
  bool isNumpyNativeType()
  {
    if((int)NumpyEquivalentType<Scalar>::type_code == NPY_USERDEF)
      return false;
    return true;
  }

  template<typename Scalar>
  bool np_type_is_convertible_into_scalar(const int np_type)
  {
    if(NumpyEquivalentType<Scalar>::type_code == np_type)
      return true;
    
    switch(np_type)
    {
      case NPY_INT:
        return FromTypeToType<int,Scalar>::value;
      case NPY_LONG:
        return FromTypeToType<long,Scalar>::value;
      case NPY_FLOAT:
        return FromTypeToType<float,Scalar>::value;
      case NPY_CFLOAT:
        return FromTypeToType<std::complex<float>,Scalar>::value;
      case NPY_DOUBLE:
        return FromTypeToType<double,Scalar>::value;
      case NPY_CDOUBLE:
        return FromTypeToType<std::complex<double>,Scalar>::value;
      case NPY_LONGDOUBLE:
        return FromTypeToType<long double,Scalar>::value;
      case NPY_CLONGDOUBLE:
        return FromTypeToType<std::complex<long double>,Scalar>::value;
      default:
        return false;
    }
  }
   
  enum NP_TYPE
  {
    MATRIX_TYPE,
    ARRAY_TYPE
  };
  
  struct EIGENPY_DLLAPI NumpyType
  {
    
    static NumpyType & getInstance();

    operator bp::object () { return getInstance().CurrentNumpyType; }

    static bp::object make(PyArrayObject* pyArray, bool copy = false);
    
    static bp::object make(PyObject* pyObj, bool copy = false);
    
    static void setNumpyType(bp::object & obj);
    
    static void sharedMemory(const bool value);
    
    static bool sharedMemory();
    
    static void switchToNumpyArray();
    
    static void switchToNumpyMatrix();
    
    static NP_TYPE & getType();
    
    static bp::object getNumpyType();
    
    static const PyTypeObject * getNumpyMatrixType();
    
    static const PyTypeObject * getNumpyArrayType();
    
    static bool isMatrix();
    
    static bool isArray();

  protected:
    
    NumpyType();

    bp::object CurrentNumpyType;
    bp::object pyModule;
    
    // Numpy types
    bp::object NumpyMatrixObject; PyTypeObject * NumpyMatrixType;
    //bp::object NumpyAsMatrixObject; PyTypeObject * NumpyAsMatrixType;
    bp::object NumpyArrayObject; PyTypeObject * NumpyArrayType;

    NP_TYPE np_type;
    
    bool shared_memory;
  };
}

#endif // ifndef __eigenpy_numpy_type_hpp__
