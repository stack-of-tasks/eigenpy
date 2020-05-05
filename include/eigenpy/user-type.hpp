//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_user_type_hpp__
#define __eigenpy_user_type_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/exception.hpp"

#include <algorithm>
#include <map>
#include <typeinfo>
#include <string>
#include <iostream>

namespace eigenpy
{
  namespace internal
  {
    template<typename T, int type_code = NumpyEquivalentType<T>::type_code>
    struct SpecialMethods
    {
      static void copyswap(void * /*dst*/, void * /*src*/, int /*swap*/, void * /*arr*/) {};
      static PyObject * getitem(void * /*ip*/, void * /*ap*/) { return NULL; };
      static int setitem(PyObject * /*op*/, void * /*ov*/, void * /*ap*/) { return -1; }
      static void copyswapn(void * /*dest*/, long /*dstride*/, void * /*src*/,
                            long /*sstride*/, long /*n*/, int /*swap*/, void * /*arr*/) {};
      static npy_bool nonzero(void * /*ip*/, void * /*array*/) { return (npy_bool)false; };
//      static void cast(void * /*from*/, void * /*to*/, npy_intp /*n*/, void * /*fromarr*/, void * /*toarr*/) {};
    };
  
    template<typename T>
    struct SpecialMethods<T,NPY_USERDEF>
    {
      static void copyswap(void * dst, void * src, int swap, void * /*arr*/)
      {
//        std::cout << "copyswap" << std::endl;
        if (src != NULL)
        {
          T & t1 = *static_cast<T*>(dst);
          T & t2 = *static_cast<T*>(src);
          t1 = t2;
        }
          
        if(swap)
        {
          T & t1 = *static_cast<T*>(dst);
          T & t2 = *static_cast<T*>(src);
          std::swap(t1,t2);
        }
      }
      
      static PyObject * getitem(void * ip, void * ap)
      {
//        std::cout << "getitem" << std::endl;
        PyArrayObject * py_array = static_cast<PyArrayObject *>(ap);
        if((py_array==NULL) || PyArray_ISBEHAVED_RO(py_array))
        {
          T * elt_ptr = static_cast<T*>(ip);
          bp::object m(boost::ref(*elt_ptr));
          Py_INCREF(m.ptr());
          return m.ptr();
        }
        else
        {
          T * elt_ptr = static_cast<T*>(ip);
          bp::object m(boost::ref(*elt_ptr));
          Py_INCREF(m.ptr());
          return m.ptr();
        }
      }
      
      static int setitem(PyObject * src_obj, void * dest_ptr, void * array)
      {
//        std::cout << "setitem" << std::endl;
        if(array == NULL)
        {
          eigenpy::Exception("Cannot retrieve the type stored in the array.");
          return -1;
        }
        PyArrayObject * py_array = static_cast<PyArrayObject *>(array);
        PyArray_Descr * descr = PyArray_DTYPE(py_array);
        PyTypeObject * array_scalar_type = descr->typeobj;
        PyTypeObject * src_obj_type = Py_TYPE(src_obj);
        
        if(array_scalar_type != src_obj_type)
        {
          return -1;
        }
        
        bp::extract<T&> extract_src_obj(src_obj);
        if(!extract_src_obj.check())
        {
          std::stringstream ss;
          ss << "The input type is of wrong type. ";
          ss << "The expected type is " << bp::type_info(typeid(T)).name() << std::endl;
          eigenpy::Exception(ss.str());
          return -1;
        }
        
        const T & src = extract_src_obj();
        T & dest = *static_cast<T*>(dest_ptr);
        dest = src;

        return 0;
      }
      
      static void copyswapn(void * dst, long dstride, void * src, long sstride,
                            long n, int swap, void * array)
      {
//        std::cout << "copyswapn" << std::endl;
        
        char *dstptr = static_cast<char*>(dst);
        char *srcptr = static_cast<char*>(src);
        
        PyArrayObject * py_array = static_cast<PyArrayObject *>(array);
        PyArray_CopySwapFunc * copyswap = PyArray_DESCR(py_array)->f->copyswap;
        
        for (npy_intp i = 0; i < n; i++)
        {
          copyswap(dstptr, srcptr, swap, array);
          dstptr += dstride;
          srcptr += sstride;
        }
      }
      
      static npy_bool nonzero(void * ip, void * array)
      {
//        std::cout << "nonzero" << std::endl;
        static const T ZeroValue = T(0);
        PyArrayObject * py_array = static_cast<PyArrayObject *>(array);
        if(py_array == NULL || PyArray_ISBEHAVED_RO(py_array))
        {
          const T & value = *static_cast<T*>(ip);
          std::cout << "value != ZeroValue: " << (bool)(value != ZeroValue) << std::endl;
          std::cout << "value != ZeroValue: " << (npy_bool)(value != ZeroValue) << std::endl;
          return (npy_bool)(value != ZeroValue);
        }
        else
        {
          T tmp_value;
          PyArray_DESCR(py_array)->f->copyswap(&tmp_value, ip, PyArray_ISBYTESWAPPED(py_array),
                                               array);
          return (npy_bool)(tmp_value != ZeroValue);
        }
      }
      
//      static void cast(void * from, void * to, npy_intp n, void * fromarr, void * toarr)
//      {
//      }

    };
  
#define EIGENPY_REGISTER_BINARY_OPERATOR(name,op) \
    template<typename T1, typename T2, typename R> \
    void binary_op_##name(char** args, npy_intp * dimensions, npy_intp * steps, void * /*data*/) \
    { \
std::cout << "binary_op_##name" << std::endl; \
      npy_intp is0 = steps[0], is1 = steps[1], \
      os = steps[2], n = *dimensions; \
      char * i0 = args[0], *i1 = args[1], *o = args[2]; \
      int k; \
      for (k = 0; k < n; k++) \
      { \
        T1 & x = *static_cast<T1*>(static_cast<void*>(i0)); \
        T2 & y = *static_cast<T2*>(static_cast<void*>(i1)); \
        R & res = *static_cast<R*>(static_cast<void*>(o)); \
        res = x op y; \
        i0 += is0; i1 += is1; o += os; \
      } \
    } \
    \
    template<typename T> \
    void binary_op_##name(char** args, npy_intp * dimensions, npy_intp * steps, void * data) \
    { \
      binary_op_##name<T,T,T>(args,dimensions,steps,data); \
    }
  
    EIGENPY_REGISTER_BINARY_OPERATOR(add,+)
    EIGENPY_REGISTER_BINARY_OPERATOR(subtract,-)
    EIGENPY_REGISTER_BINARY_OPERATOR(multiply,*)
    EIGENPY_REGISTER_BINARY_OPERATOR(divide,/)
    EIGENPY_REGISTER_BINARY_OPERATOR(equal,==)
    EIGENPY_REGISTER_BINARY_OPERATOR(not_equal,!=)
    EIGENPY_REGISTER_BINARY_OPERATOR(less,<)
    EIGENPY_REGISTER_BINARY_OPERATOR(greater,>)
    EIGENPY_REGISTER_BINARY_OPERATOR(less_equal,<=)
    EIGENPY_REGISTER_BINARY_OPERATOR(greater_equal,>=)
  
  } // namespace internal

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
  
  std::cout << "type name: " << py_type->tp_name << std::endl;
  std::cout << "code: " << code << std::endl;
  std::cout << "PyArray_TypeNum(py_type): " << PyArray_TypeNum(py_type) << std::endl;
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
                               PyArray_CopySwapNFunc * copyswapn)
    {
      namespace bp = boost::python;
  
      PyArray_Descr * descr_ptr = new PyArray_Descr(*PyArray_DescrFromType(NPY_OBJECT));
      PyArray_Descr & descr = *descr_ptr;
      descr.typeobj = py_type_ptr;
      descr.kind = 'V';
      descr.byteorder = '=';
      std::cout << "type name: " << py_type_ptr->tp_name << std::endl;
      descr.elsize = type_size;
      descr.flags = NPY_LIST_PICKLE | NPY_USE_GETITEM | NPY_USE_SETITEM | NPY_NEEDS_INIT | NPY_NEEDS_PYAPI;
//      descr->names = PyTuple_New(0);
//      descr->fields = PyDict_New();
      
      std::cout << "descr->elsize: " << descr.elsize << std::endl;
//      std::cout << "size of Scalar: " << sizeof(Scalar) << std::endl;
      std::cout << "type_num: " << descr.type_num << std::endl;
      
      PyArray_ArrFuncs * funcs_ptr = new PyArray_ArrFuncs;
      PyArray_ArrFuncs & funcs = *funcs_ptr;
      descr.f = funcs_ptr;
      PyArray_InitArrFuncs(funcs_ptr);
      funcs.getitem = getitem;
      funcs.setitem = setitem;
      funcs.nonzero = nonzero;
      funcs.copyswap = copyswap;
      funcs.copyswapn = copyswapn;
//      f->cast = cast;

      const int code = PyArray_RegisterDataType(descr_ptr);
      assert(code >= 0 && "The return code should be positive");
      PyArray_Descr * new_descr = PyArray_DescrFromType(code);
      std::cout << "new type_num: " << new_descr->type_num << std::endl;
  
      type_to_py_type_bindings.insert(std::make_pair(type_info_ptr,py_type_ptr));
      py_array_descr_bindings[py_type_ptr] = new_descr;
      py_array_code_bindings[py_type_ptr] = code;
      
//      PyArray_RegisterCanCast(descr,NPY_OBJECT,NPY_NOSCALAR);
      return code;
    }
    
    static Register & instance()
    {
      return self;
    }
    
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
    
    static Register self;
    
  };

  template<typename Scalar>
  int registerNewType(PyTypeObject * py_type_ptr = NULL)
  {
    // Check whether the type is a Numpy native type.
    // In this case, the registration is not required.
    if(isNumpyNativeType<Scalar>())
      return NumpyEquivalentType<Scalar>::type_code;
    
    // Retrieve the registered type for the current Scalar
    if(py_type_ptr == NULL)
    { // retrive the type from Boost.Python
      py_type_ptr = Register::getPyType<Scalar>();
    }
    
    if(Register::isRegistered(py_type_ptr))
      return Register::getTypeCode(py_type_ptr); // the type is already registered
    
    PyArray_GetItemFunc * getitem = &internal::SpecialMethods<Scalar>::getitem;
    PyArray_SetItemFunc * setitem = &internal::SpecialMethods<Scalar>::setitem;
    PyArray_NonzeroFunc * nonzero = &internal::SpecialMethods<Scalar>::nonzero;
    PyArray_CopySwapFunc * copyswap = &internal::SpecialMethods<Scalar>::copyswap;
    PyArray_CopySwapNFunc * copyswapn = &internal::SpecialMethods<Scalar>::copyswapn;
//    PyArray_CastFunc * cast = &internal::SpecialMethods<Scalar>::cast;
    
    int code =  Register::registerNewType(py_type_ptr,
                                          &typeid(Scalar),
                                          sizeof(Scalar),
                                          getitem, setitem, nonzero,
                                          copyswap, copyswapn);
    
    std::cout << "code: " << code << std::endl;
    PyObject* numpy_str;
#if PY_MAJOR_VERSION >= 3
    numpy_str = PyUnicode_FromString("numpy");
#else
    numpy_str = PyString_FromString("numpy");
#endif
    PyObject* numpy;
    numpy = PyImport_Import(numpy_str);
    Py_DECREF(numpy_str);
    
    // load numpy
    std::cout << "numpy loaded" << std::endl;
    import_ufunc();
    #define REGISTER_BINARY_UFUNC(name,T1,T2,R) { \
         PyUFuncObject* ufunc = \
             (PyUFuncObject*)PyObject_GetAttrString(numpy, #name); \
std::cout << "ufunc" << std::endl; \
         int _types[3] = { Register::getTypeCode<T1>(), Register::getTypeCode<T2>(), Register::getTypeCode<R>()}; \
         if (!ufunc) { \
         std::cout << "ufunc not available" << std::endl; \
             /*goto fail; \*/ \
         } \
         std::cout << "step 1" << std::endl; \
         if (sizeof(_types)/sizeof(int)!=ufunc->nargs) { \
             PyErr_Format(PyExc_AssertionError, \
                          "ufunc %s takes %d arguments, our loop takes %lu", \
                          #name, ufunc->nargs, (unsigned long) \
                          (sizeof(_types)/sizeof(int))); \
             Py_DECREF(ufunc); \
std::cout << "failed 1" << std::endl; \
         } \
         std::cout << "step 2" << std::endl; \
         if (PyUFunc_RegisterLoopForType((PyUFuncObject*)ufunc, code, \
              internal::binary_op_##name<T1,T2,R>, _types, 0) < 0) { \
             /*Py_DECREF(ufunc);*/ \
             std::cout << "failed 2" << std::endl; \
             /*goto fail; \*/ \
         } \
         Py_DECREF(ufunc); \
     }

    std::cout << "register func" << std::endl;
    REGISTER_BINARY_UFUNC(add,Scalar,Scalar,Scalar);
    REGISTER_BINARY_UFUNC(subtract,Scalar,Scalar,Scalar);
    REGISTER_BINARY_UFUNC(multiply,Scalar,Scalar,Scalar);
    REGISTER_BINARY_UFUNC(divide,Scalar,Scalar,Scalar);
  
    // Comparison operators
    REGISTER_BINARY_UFUNC(equal,Scalar,Scalar,bool);
    REGISTER_BINARY_UFUNC(not_equal,Scalar,Scalar,bool);
    REGISTER_BINARY_UFUNC(greater,Scalar,Scalar,bool);
    REGISTER_BINARY_UFUNC(less,Scalar,Scalar,bool);
    REGISTER_BINARY_UFUNC(greater_equal,Scalar,Scalar,bool);
    REGISTER_BINARY_UFUNC(less_equal,Scalar,Scalar,bool);
    std::cout << "register func" << std::endl;
    
    Py_DECREF(numpy);
    
    return code;
  }
  
} // namespace eigenpy

#endif // __eigenpy_user_type_hpp__
