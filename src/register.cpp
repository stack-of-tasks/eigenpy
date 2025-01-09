/*
 * Copyright 2020-2021 INRIA
 */

#include "eigenpy/register.hpp"

namespace eigenpy {

PyArray_Descr* Register::getPyArrayDescr(PyTypeObject* py_type_ptr) {
  MapDescr::iterator it = instance().py_array_descr_bindings.find(py_type_ptr);
  if (it != instance().py_array_descr_bindings.end())
    return it->second;
  else
    return NULL;
}

PyArray_Descr* Register::getPyArrayDescrFromTypeNum(const int type_num) {
  if (type_num >= NPY_USERDEF) {
    for (const auto& elt : instance().py_array_code_bindings) {
      if (elt.second == type_num)
        return instance().py_array_descr_bindings[elt.first];
    }
    return nullptr;
  } else
    return PyArray_DescrFromType(type_num);
}

bool Register::isRegistered(PyTypeObject* py_type_ptr) {
  if (getPyArrayDescr(py_type_ptr) != NULL)
    return true;
  else
    return false;
}

int Register::getTypeCode(PyTypeObject* py_type_ptr) {
  MapCode::iterator it = instance().py_array_code_bindings.find(py_type_ptr);
  if (it != instance().py_array_code_bindings.end())
    return it->second;
  else
    return PyArray_TypeNum(py_type_ptr);
}

int Register::registerNewType(
    PyTypeObject* py_type_ptr, const std::type_info* type_info_ptr,
    const int type_size, const int alignement, PyArray_GetItemFunc* getitem,
    PyArray_SetItemFunc* setitem, PyArray_NonzeroFunc* nonzero,
    PyArray_CopySwapFunc* copyswap, PyArray_CopySwapNFunc* copyswapn,
    PyArray_DotFunc* dotfunc, PyArray_FillFunc* fill,
    PyArray_FillWithScalarFunc* fillwithscalar) {
  bp::tuple tp_bases_extended(
      bp::make_tuple(bp::handle<>(bp::borrowed(&PyGenericArrType_Type))));
  tp_bases_extended +=
      bp::tuple(bp::handle<>(bp::borrowed(py_type_ptr->tp_bases)));

  Py_INCREF(tp_bases_extended.ptr());
  py_type_ptr->tp_bases = tp_bases_extended.ptr();

  py_type_ptr->tp_flags &= ~Py_TPFLAGS_READY;  // to force the rebuild
  //    py_type_ptr->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
  if (PyType_Ready(py_type_ptr) <
      0)  // Force rebuilding of the __bases__ and mro
  {
    throw std::invalid_argument("PyType_Ready fails to initialize input type.");
  }

  PyArray_DescrProto* descr_ptr = new PyArray_DescrProto();
  PyArray_DescrProto& descr = *descr_ptr;
  descr.typeobj = py_type_ptr;
  descr.kind = 'V';
  descr.byteorder = '=';
  descr.type = 'r';
  descr.elsize = type_size;
  descr.flags =
      NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT;
  descr.type_num = 0;
  descr.names = 0;
  descr.fields = 0;
  descr.alignment =
      alignement;  // call_PyArray_DescrFromType(NPY_OBJECT)->alignment;

  PyArray_ArrFuncs* funcs_ptr = new PyArray_ArrFuncs;
  PyArray_ArrFuncs& funcs = *funcs_ptr;
  descr.f = funcs_ptr;
  call_PyArray_InitArrFuncs(funcs_ptr);
  funcs.getitem = getitem;
  funcs.setitem = setitem;
  funcs.nonzero = nonzero;
  funcs.copyswap = copyswap;
  funcs.copyswapn = copyswapn;
  funcs.dotfunc = dotfunc;
  funcs.fill = fill;
  funcs.fillwithscalar = fillwithscalar;
  //      f->cast = cast;
  Py_SET_TYPE(descr_ptr, &PyArrayDescr_Type);

  const int code = call_PyArray_RegisterDataType(descr_ptr);
  assert(code >= 0 && "The return code should be positive");
  PyArray_Descr* new_descr = call_PyArray_DescrFromType(code);

  if (PyDict_SetItemString(py_type_ptr->tp_dict, "dtype",
                           (PyObject*)new_descr) < 0) {
    throw std::invalid_argument("PyDict_SetItemString fails.");
  }

  instance().type_to_py_type_bindings.insert(
      std::make_pair(type_info_ptr, py_type_ptr));
  instance().py_array_descr_bindings[py_type_ptr] = new_descr;
  instance().py_array_code_bindings[py_type_ptr] = code;

  //      PyArray_RegisterCanCast(descr,NPY_OBJECT,NPY_NOSCALAR);
  return code;
}

Register& Register::instance() {
  static Register self;
  return self;
}

}  // namespace eigenpy
