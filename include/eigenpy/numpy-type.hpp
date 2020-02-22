/*
 * Copyright 2018-2020 INRIA
*/

#ifndef __eigenpy_numpy_type_hpp__
#define __eigenpy_numpy_type_hpp__

#include "eigenpy/fwd.hpp"

#include <patchlevel.h> // For PY_MAJOR_VERSION

namespace eigenpy
{
  namespace bp = boost::python;

  template <typename SCALAR>  struct NumpyEquivalentType {};

  template <> struct NumpyEquivalentType<float>   { enum { type_code = NPY_FLOAT  };};
  template <> struct NumpyEquivalentType< std::complex<float> >   { enum { type_code = NPY_CFLOAT  };};
  template <> struct NumpyEquivalentType<double>  { enum { type_code = NPY_DOUBLE };};
  template <> struct NumpyEquivalentType< std::complex<double> >  { enum { type_code = NPY_CDOUBLE };};
  template <> struct NumpyEquivalentType<long double>  { enum { type_code = NPY_LONGDOUBLE };};
  template <> struct NumpyEquivalentType< std::complex<long double> >  { enum { type_code = NPY_CLONGDOUBLE };};
  template <> struct NumpyEquivalentType<int>     { enum { type_code = NPY_INT    };};
  template <> struct NumpyEquivalentType<long>    { enum { type_code = NPY_LONG    };};
   
  enum NP_TYPE
  {
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

    operator bp::object () { return getInstance().CurrentNumpyType; }

    static bp::object make(PyArrayObject* pyArray, bool copy = false)
    { return make((PyObject*)pyArray,copy); }
    
    static bp::object make(PyObject* pyObj, bool copy = false)
    {
      bp::object m;
      if(isMatrix())
        m = getInstance().NumpyMatrixObject(bp::object(bp::handle<>(pyObj)), bp::object(), copy);
//        m = NumpyAsMatrixObject(bp::object(bp::handle<>(pyObj)));
      else if(isArray())
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
      getInstance().getType() = ARRAY_TYPE;
    }
    
    static void switchToNumpyMatrix()
    {
      getInstance().CurrentNumpyType = getInstance().NumpyMatrixObject;
      getInstance().getType() = MATRIX_TYPE;
    }
    
    static NP_TYPE & getType()
    {
      return getInstance().np_type;
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
      if(getInstance().isMatrix()) return false;
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
      
      CurrentNumpyType = NumpyArrayObject; // default conversion
      np_type = ARRAY_TYPE;
    }

    bp::object CurrentNumpyType;
    bp::object pyModule;
    
    // Numpy types
    bp::object NumpyMatrixObject; PyTypeObject * NumpyMatrixType;
    //bp::object NumpyAsMatrixObject; PyTypeObject * NumpyAsMatrixType;
    bp::object NumpyArrayObject; PyTypeObject * NumpyArrayType;

    NP_TYPE np_type;
  };
}

#endif // ifndef __eigenpy_numpy_type_hpp__
