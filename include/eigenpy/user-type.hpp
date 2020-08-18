//
// Copyright (c) 2020 INRIA
//

#ifndef __eigenpy_user_type_hpp__
#define __eigenpy_user_type_hpp__

#include "eigenpy/fwd.hpp"
#include "eigenpy/numpy-type.hpp"
#include "eigenpy/register.hpp"

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
      static void dotfunc(void * /*ip0_*/, npy_intp /*is0*/, void * /*ip1_*/, npy_intp /*is1*/,
                          void * /*op*/, npy_intp /*n*/, void * /*arr*/);
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
      
      static void dotfunc(void * ip0_, npy_intp is0, void * ip1_, npy_intp is1,
                          void * op, npy_intp n, void * /*arr*/)
      {
          T res = T(0);
          char *ip0 = (char*)ip0_, *ip1 = (char*)ip1_;
          npy_intp i;
          for(i = 0; i < n; i++)
          {
            
            res += *static_cast<T*>(static_cast<void*>(ip0))
            * *static_cast<T*>(static_cast<void*>(ip1));
            ip0 += is0;
            ip1 += is1;
          }
          *static_cast<T*>(op) = res;
      }
      
//      static void cast(void * from, void * to, npy_intp n, void * fromarr, void * toarr)
//      {
//      }

    };
  
  } // namespace internal

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
    PyArray_CopySwapNFunc * copyswapn = reinterpret_cast<PyArray_CopySwapNFunc*>(&internal::SpecialMethods<Scalar>::copyswapn);
    PyArray_DotFunc * dotfunc = &internal::SpecialMethods<Scalar>::dotfunc;
//    PyArray_CastFunc * cast = &internal::SpecialMethods<Scalar>::cast;
    
    int code = Register::registerNewType(py_type_ptr,
                                         &typeid(Scalar),
                                         sizeof(Scalar),
                                         getitem, setitem, nonzero,
                                         copyswap, copyswapn,
                                         dotfunc);
    
    return code;
  }
  
} // namespace eigenpy

#endif // __eigenpy_user_type_hpp__
