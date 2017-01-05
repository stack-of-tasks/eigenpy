/*
 * Copyright 2014, Nicolas Mansard, LAAS-CNRS
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

#ifndef __eigenpy_details_hpp__
#define __eigenpy_details_hpp__

#include <boost/python.hpp>
#include <Eigen/Core>

#include <numpy/numpyconfig.h>
#ifdef NPY_1_8_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>
#include <iostream>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/registration.hpp"
#include "eigenpy/exception.hpp"
#include "eigenpy/map.hpp"


namespace eigenpy
{
  template <typename SCALAR>  struct NumpyEquivalentType {};
  template <> struct NumpyEquivalentType<double>  { enum { type_code = NPY_DOUBLE };};
  template <> struct NumpyEquivalentType<int>     { enum { type_code = NPY_INT    };};
  template <> struct NumpyEquivalentType<float>   { enum { type_code = NPY_FLOAT  };};

  namespace bp = boost::python;

  struct PyMatrixType
  {

    static PyMatrixType & getInstance()
    {
      static PyMatrixType instance;
      return instance;
    }

    operator bp::object () { return pyMatrixType; }

    bp::object make(PyArrayObject* pyArray, bool copy = false)
    { return make((PyObject*)pyArray,copy); }
    bp::object make(PyObject* pyObj, bool copy = false)
    {
      boost::python::object m
      = pyMatrixType(bp::object(bp::handle<>(pyObj)), bp::object(), copy);
      Py_INCREF(m.ptr());
      return m;
    }

  protected:
    PyMatrixType()
    {
      pyModule = boost::python::import("numpy");
      pyMatrixType = pyModule.attr("matrix");
    }

    bp::object pyMatrixType;
    bp::object pyModule;
  };

  /* --- TO PYTHON -------------------------------------------------------------- */
  template< typename MatType,typename EquivalentEigenType >
  struct EigenToPy
  {
    static PyObject* convert(MatType const& mat)
    {
      typedef typename MatType::Scalar T;
      assert( (mat.rows()<INT_MAX) && (mat.cols()<INT_MAX) 
	      && "Matrix range larger than int ... should never happen." );
      const int R  = (int)mat.rows(), C = (int)mat.cols();

      npy_intp shape[2] = { R,C };
      PyArrayObject* pyArray = (PyArrayObject*)
	PyArray_SimpleNew(2, shape, NumpyEquivalentType<T>::type_code);

      MapNumpy<EquivalentEigenType>::map(pyArray) = mat;

      return PyMatrixType::getInstance().make(pyArray).ptr();
    }
  };
  
  /* --- FROM PYTHON ------------------------------------------------------------ */
  namespace bp = boost::python;

  template<typename MatType, int ROWS,int COLS>
  struct TraitsMatrixConstructor
  {
    static MatType & construct(void*storage,int /*r*/,int /*c*/)
    {
      return * new(storage) MatType();
    }
  };

  template<typename MatType>
  struct TraitsMatrixConstructor<MatType,Eigen::Dynamic,Eigen::Dynamic>
  {
    static MatType & construct(void*storage,int r,int c)
    {
      return * new(storage) MatType(r,c);
    }
  };

  template<typename MatType,int R>
  struct TraitsMatrixConstructor<MatType,R,Eigen::Dynamic>
  {
    static MatType & construct(void*storage,int /*r*/,int c)
    {
      return * new(storage) MatType(R,c);
    }
  };

  template<typename MatType,int C>
  struct TraitsMatrixConstructor<MatType,Eigen::Dynamic,C>
  {
    static MatType & construct(void*storage,int r,int /*c*/)
    {
      return * new(storage) MatType(r,C);
    }
  };


  template<typename MatType,typename EquivalentEigenType>
  struct EigenFromPy
  {
    EigenFromPy()
    {
      bp::converter::registry::push_back
	(reinterpret_cast<void *(*)(_object *)>(&convertible),
	 &construct,bp::type_id<MatType>());
    }
 
    // Determine if obj_ptr can be converted in a Eigenvec
    static void* convertible(PyArrayObject* obj_ptr)
    {
      typedef typename MatType::Scalar T;

      if (!PyArray_Check(obj_ptr)) 
	{
#ifndef NDEBUG
	  std::cerr << "The python object is not a numpy array." << std::endl;
#endif
	  return 0;
	}

      if (PyArray_NDIM(obj_ptr) != 2)
	if ( (PyArray_NDIM(obj_ptr) !=1) || (! MatType::IsVectorAtCompileTime) )
	  {
#ifndef NDEBUG
	    std::cerr << "The number of dimension of the object is not correct." << std::endl;
#endif
	    return 0;
	  }

      if ((PyArray_ObjectType(reinterpret_cast<PyObject *>(obj_ptr), 0)) != NumpyEquivalentType<T>::type_code)
	{
#ifndef NDEBUG
	  std::cerr << "The internal type as no Eigen equivalent." << std::endl;
#endif
	  return 0;
	}
#ifdef NPY_1_8_API_VERSION
      if (!(PyArray_FLAGS(obj_ptr)))
#else
      if (!(PyArray_FLAGS(obj_ptr) & NPY_ALIGNED))
#endif
	{
#ifndef NDEBUG
	  std::cerr << "NPY non-aligned matrices are not implemented." << std::endl;
#endif
	  return 0;
	}
      
      return obj_ptr;
    }
 
    // Convert obj_ptr into a Eigenvec
    static void construct(PyObject* pyObj,
			  bp::converter::rvalue_from_python_stage1_data* memory)
    {
      using namespace Eigen;

      PyArrayObject * pyArray = reinterpret_cast<PyArrayObject*>(pyObj);
      typename MapNumpy<EquivalentEigenType>::EigenMap numpyMap = MapNumpy<EquivalentEigenType>::map(pyArray);

      void* storage = ((bp::converter::rvalue_from_python_storage<MatType>*)
		       ((void*)memory))->storage.bytes;
      assert( (numpyMap.rows()<INT_MAX) && (numpyMap.cols()<INT_MAX) 
	      && "Map range larger than int ... can never happen." );
      int r=(int)numpyMap.rows(),c=(int)numpyMap.cols();
      EquivalentEigenType & eigenMatrix = //* new(storage) MatType(numpyMap.rows(),numpyMap.cols());
	TraitsMatrixConstructor<MatType,MatType::RowsAtCompileTime,MatType::ColsAtCompileTime>::construct (storage,r,c);
      memory->convertible = storage;

      eigenMatrix = numpyMap;
    }
  };
#define numpy_import_array() {if (_import_array() < 0) {PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); } }
  
  template<typename MatType,typename EigenEquivalentType>
  void enableEigenPySpecific()
  {
    numpy_import_array();
    if(check_registration<MatType>()) return;
    
    boost::python::to_python_converter<MatType,EigenToPy<MatType,MatType> >();
    EigenFromPy<MatType,MatType>();
  }

} // namespace eigenpy

#endif // ifndef __eigenpy_details_hpp__
