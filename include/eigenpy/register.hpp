//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_register_hpp__
#define __eigenpy_register_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/exception.hpp"

#include <algorithm>
#include <map>
#include <typeinfo>
#include <string>

namespace eigenpy
{

  /// \brief Structure collecting all the types registers in Numpy via EigenPy
  struct EIGENPY_DLLEXPORT Register
  {
    
    static PyArray_Descr * getPyArrayDescr(PyTypeObject * py_type_ptr)
    {
      MapDescr::iterator it = py_array_descr_bindings.find(py_type_ptr);
      if(it != py_array_descr_bindings.end())
        return it->second;
      else
        return NULL;
    }
    
    template<typename Scalar>
    static bool isRegistered()
    {
      return isRegistered(Register::getPyType<Scalar>());
    }
    
    static bool isRegistered(PyTypeObject * py_type_ptr)
    {
      if(getPyArrayDescr(py_type_ptr) != NULL)
        return true;
      else
        return false;
    }
    
    static int getTypeCode(PyTypeObject * py_type_ptr)
    {
      MapCode::iterator it = py_array_code_bindings.find(py_type_ptr);
      if(it != py_array_code_bindings.end())
        return it->second;
      else
        return PyArray_TypeNum(py_type_ptr);
    }
  
    template<typename Scalar>
    static PyTypeObject * getPyType()
    {
      if(!isNumpyNativeType<Scalar>())
      {
        const PyTypeObject * const_py_type_ptr = bp::converter::registered_pytype<Scalar>::get_pytype();
        if(const_py_type_ptr == NULL)
        {
          std::stringstream ss;
          ss << "The type " << typeid(Scalar).name() << " does not have a registered converter inside Boot.Python." << std::endl;
          throw std::invalid_argument(ss.str());
        }
        PyTypeObject * py_type_ptr = const_cast<PyTypeObject *>(const_py_type_ptr);
        return py_type_ptr;
      }
      else
      {
        PyArray_Descr * new_descr = PyArray_DescrFromType(NumpyEquivalentType<Scalar>::type_code);
        return new_descr->typeobj;
      }
    }

    template<typename Scalar>
    static int getTypeCode()
    {
      if(isNumpyNativeType<Scalar>())
        return NumpyEquivalentType<Scalar>::type_code;
      else
      {
        const std::type_info & info = typeid(Scalar);
        if(type_to_py_type_bindings.find(&info) != type_to_py_type_bindings.end())
        {
          PyTypeObject * py_type = type_to_py_type_bindings[&info];
          int code = py_array_code_bindings[py_type];
  
          return code;
        }
        else
          return -1; // type not registered
      }
    }
    
    static int registerNewType(PyTypeObject * py_type_ptr,
                               const std::type_info * type_info_ptr,
                               const int type_size,
                               PyArray_GetItemFunc * getitem,
                               PyArray_SetItemFunc * setitem,
                               PyArray_NonzeroFunc * nonzero,
                               PyArray_CopySwapFunc * copyswap,
                               PyArray_CopySwapNFunc * copyswapn,
                               PyArray_DotFunc * dotfunc)
    {
      namespace bp = boost::python;
  
      PyArray_Descr * descr_ptr = new PyArray_Descr(*PyArray_DescrFromType(NPY_OBJECT));
      PyArray_Descr & descr = *descr_ptr;
      descr.typeobj = py_type_ptr;
      descr.kind = 'V';
      descr.byteorder = '=';
      descr.elsize = type_size;
      descr.flags = NPY_LIST_PICKLE | NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT | NPY_NEEDS_PYAPI;
//      descr->names = PyTuple_New(0);
//      descr->fields = PyDict_New();
      
      PyArray_ArrFuncs * funcs_ptr = new PyArray_ArrFuncs;
      PyArray_ArrFuncs & funcs = *funcs_ptr;
      descr.f = funcs_ptr;
      PyArray_InitArrFuncs(funcs_ptr);
      funcs.getitem = getitem;
      funcs.setitem = setitem;
      funcs.nonzero = nonzero;
      funcs.copyswap = copyswap;
      funcs.copyswapn = copyswapn;
      funcs.dotfunc = dotfunc;
//      f->cast = cast;

      const int code = PyArray_RegisterDataType(descr_ptr);
      assert(code >= 0 && "The return code should be positive");
      PyArray_Descr * new_descr = PyArray_DescrFromType(code);

      type_to_py_type_bindings.insert(std::make_pair(type_info_ptr,py_type_ptr));
      py_array_descr_bindings[py_type_ptr] = new_descr;
      py_array_code_bindings[py_type_ptr] = code;
      
//      PyArray_RegisterCanCast(descr,NPY_OBJECT,NPY_NOSCALAR);
      return code;
    }
    
//    static Register & instance()
//    {
//      return self;
//    }
    
  private:
    
    Register() {};
    
    struct Compare_PyTypeObject
    {
      bool operator()(const PyTypeObject * a, const PyTypeObject * b) const
      {
        return std::string(a->tp_name) < std::string(b->tp_name);
      }
    };
  
    struct Compare_TypeInfo
    {
      bool operator()(const std::type_info * a, const std::type_info * b) const
      {
        return std::string(a->name()) < std::string(b->name());
      }
    };
  
    typedef std::map<const std::type_info *,PyTypeObject *,Compare_TypeInfo> MapInfo;
    static MapInfo type_to_py_type_bindings;
    
    typedef std::map<PyTypeObject *,PyArray_Descr *,Compare_PyTypeObject> MapDescr;
    static MapDescr py_array_descr_bindings;
    
    typedef std::map<PyTypeObject *,int,Compare_PyTypeObject> MapCode;
    static MapCode py_array_code_bindings;
    
//    static Register self;
    
  };
  
} // namespace eigenpy

#endif // __eigenpy_register_hpp__
