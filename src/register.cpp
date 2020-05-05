/*
 * Copyright 2020 INRIA
 */

#include "eigenpy/register.hpp"

namespace eigenpy
{

  PyArray_Descr * Register::getPyArrayDescr(PyTypeObject * py_type_ptr)
  {
    MapDescr::iterator it = instance().py_array_descr_bindings.find(py_type_ptr);
    if(it != instance().py_array_descr_bindings.end())
      return it->second;
    else
      return NULL;
  }
  
  bool Register::isRegistered(PyTypeObject * py_type_ptr)
  {
    if(getPyArrayDescr(py_type_ptr) != NULL)
      return true;
    else
      return false;
  }
  
  int Register::getTypeCode(PyTypeObject * py_type_ptr)
  {
    MapCode::iterator it = instance().py_array_code_bindings.find(py_type_ptr);
    if(it != instance().py_array_code_bindings.end())
      return it->second;
    else
      return PyArray_TypeNum(py_type_ptr);
  }

  int Register::registerNewType(PyTypeObject * py_type_ptr,
                                const std::type_info * type_info_ptr,
                                const int type_size,
                                PyArray_GetItemFunc * getitem,
                                PyArray_SetItemFunc * setitem,
                                PyArray_NonzeroFunc * nonzero,
                                PyArray_CopySwapFunc * copyswap,
                                PyArray_CopySwapNFunc * copyswapn,
                                PyArray_DotFunc * dotfunc)
  {
    PyArray_Descr * descr_ptr = new PyArray_Descr(*call_PyArray_DescrFromType(NPY_OBJECT));
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
    call_PyArray_InitArrFuncs(funcs_ptr);
    funcs.getitem = getitem;
    funcs.setitem = setitem;
    funcs.nonzero = nonzero;
    funcs.copyswap = copyswap;
    funcs.copyswapn = copyswapn;
    funcs.dotfunc = dotfunc;
    //      f->cast = cast;
    
    const int code = call_PyArray_RegisterDataType(descr_ptr);
    assert(code >= 0 && "The return code should be positive");
    PyArray_Descr * new_descr = call_PyArray_DescrFromType(code);
    
    instance().type_to_py_type_bindings.insert(std::make_pair(type_info_ptr,py_type_ptr));
    instance().py_array_descr_bindings[py_type_ptr] = new_descr;
    instance().py_array_code_bindings[py_type_ptr] = code;
    
    //      PyArray_RegisterCanCast(descr,NPY_OBJECT,NPY_NOSCALAR);
    return code;
  }

  Register & Register::instance()
  {
    static Register self;
    return self;
  }

} // namespace eigenpy
