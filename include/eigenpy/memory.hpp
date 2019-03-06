/*
 * Copyright 2014-2019, CNRS
 * Copyright 2018-2019, INRIA
 */

#ifndef __eigenpy_memory_hpp__
#define __eigenpy_memory_hpp__

#include <boost/python.hpp>

/**
 * This section contains a convenience MACRO which allows an easy specialization of
 * Boost Python Object allocator for struct data types containing Eigen objects and requiring
 * strict alignment.
 *
 * This code was proposed as an stackoverflow answer:
 *     http://stackoverflow.com/questions/13177573/how-to-expose-aligned-class-with-boost-python/29694518
 * Leading to this page proposing the solution:
 *     http://fhtagn.net/prog/2015/04/16/quaternion_boost_python.html
 *
 */
#define EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(...) \
namespace boost { namespace python { namespace objects { \
      template<> \
      struct instance< value_holder<__VA_ARGS__> > \
      { \
        typedef value_holder<__VA_ARGS__> Data; \
        PyObject_VAR_HEAD \
        PyObject* dict; \
        PyObject* weakrefs; \
        instance_holder* objects; \
         \
        typedef type_with_alignment< \
        ::boost::alignment_of<Data>::value  \
        >::type align_t; \
         \
        union \
        { \
          align_t align; \
          char bytes[sizeof(Data) + 16]; \
        } storage; \
      }; \
       \
      template<class Derived> \
      struct make_instance_impl<__VA_ARGS__, value_holder<__VA_ARGS__>, Derived> \
      { \
        typedef __VA_ARGS__ T; \
        typedef value_holder<__VA_ARGS__> Holder; \
        typedef objects::instance<Holder> instance_t; \
         \
        template <class Arg> \
        static inline PyObject* execute(Arg & x) \
        { \
          BOOST_MPL_ASSERT((mpl::or_<is_class<T>, is_union<T> >)); \
           \
          PyTypeObject* type = Derived::get_class_object(x); \
           \
          if (type == 0) \
            return python::detail::none(); \
           \
          PyObject* raw_result = type->tp_alloc(type, objects::additional_instance_size<Holder>::value); \
          if (raw_result != 0) \
          { \
            python::detail::decref_guard protect(raw_result); \
            instance_t* instance = (instance_t*)(void*)raw_result; \
            Holder* holder = Derived::construct(&instance->storage, (PyObject*)instance, x); \
            holder->install(raw_result); \
             \
            Py_ssize_t holder_offset = reinterpret_cast<Py_ssize_t>(holder) \
            - reinterpret_cast<Py_ssize_t>(&instance->storage) \
            + static_cast<Py_ssize_t>(offsetof(instance_t, storage)); \
            Py_SIZE(instance) = holder_offset; \
             \
            protect.cancel(); \
          } \
          return raw_result; \
        } \
      }; \
       \
      template<> \
      struct make_instance<__VA_ARGS__, value_holder<__VA_ARGS__> > \
      : make_instance_impl<__VA_ARGS__, value_holder<__VA_ARGS__>, make_instance<__VA_ARGS__,value_holder<__VA_ARGS__> > > \
      { \
        template <class U> \
        static inline PyTypeObject* get_class_object(U &) \
        { \
          return converter::registered<__VA_ARGS__>::converters.get_class_object(); \
        } \
         \
        static inline value_holder<__VA_ARGS__>* construct(void* storage, PyObject* instance, reference_wrapper<__VA_ARGS__ const> x) \
        { \
          void* aligned_storage = reinterpret_cast<void*>((reinterpret_cast<size_t>(storage) & ~(size_t(15))) + 16); \
          value_holder<__VA_ARGS__>* new_holder = new (aligned_storage) value_holder<__VA_ARGS__>(instance, x); \
          return new_holder; \
        } \
      }; \
    }}}

#endif // __eigenpy_memory_hpp__
